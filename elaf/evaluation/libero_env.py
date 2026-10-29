"""Policy-independent LIBERO worker. One benchmark per spawned Python process."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import traceback
import time
from collections import deque
import numpy as np

ROOT = Path('/data/users/jiachangzhang/eb_vjepa')
BASES = ('libero_spatial','libero_object','libero_goal','libero_10')


def make_manifest(benchmark, manifest_path=None, root=ROOT):
    if benchmark == 'clean':
        return [dict(benchmark='clean',suite=suite,task_id=task,trial=trial,
                     seed=239+(suite_id*10+task)*5+trial,env_seed=239+(suite_id*10+task)*5)
                for suite_id,suite in enumerate(BASES) for task in range(10) for trial in range(5)]
    if benchmark not in ('plus','pro'):
        raise ValueError(f'Unknown benchmark {benchmark}')
    path = Path(manifest_path) if manifest_path else Path(root)/'experiments/action_only'/(
        'libero_20k_plus_280_shared/manifest.json' if benchmark=='plus' else 'libero_10k_pro/manifest.json')
    document=json.loads(path.read_text())
    if document['benchmark'] != benchmark:
        raise ValueError('Manifest benchmark mismatch')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    return [dict(entry,benchmark=benchmark,trial=trial,seed=document['seed']+entry['index']*5+trial,
                 env_seed=document['seed']+entry['index']*5,manifest_path=str(path),manifest_sha256=digest)
            for entry in document['entries'] for trial in range(entry['trials'])]


def horizon(suite):
    if suite.startswith('libero_10') or suite=='libero_90':
        return 700
    if any(suite==base or suite.startswith(base+'_') for base in BASES[:3]):
        return 400
    raise ValueError(f'Unsupported suite {suite}')


def convert_actions(actions, convention='open_01'):
    value=np.asarray(actions,dtype=np.float32).copy()
    if value.ndim!=2 or value.shape[1]!=7 or not np.isfinite(value).all():
        raise ValueError('Actions must be finite [K,7] physical actions after denormalization')
    if convention=='open_01':
        value[:,-1]=np.sign(1-2*value[:,-1])
    elif convention!='simulator':
        raise ValueError('Unknown gripper convention')
    return value


def observation(obs,instruction):
    q=np.asarray(obs['robot0_eef_quat']).copy()
    q[3]=np.clip(q[3],-1,1)
    denominator=np.sqrt(1-q[3]**2)
    axis=np.zeros(3) if np.isclose(denominator,0) else q[:3]*2*np.arccos(q[3])/denominator
    return {'image':np.ascontiguousarray(obs['agentview_image'][::-1,::-1]),
            'wrist_image':np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1]),
            'proprio':np.concatenate([obs['robot0_eef_pos'],axis,obs['robot0_gripper_qpos']]).astype(np.float32),
            'instruction':instruction}


class LegacyBinaryNumpy:
    def __getattr__(self,name):
        return np.float64 if name=='float_' else getattr(np,name)
    def fromstring(self,string,dtype=float,count=-1,sep='',**kwargs):
        return np.frombuffer(string,dtype=dtype,count=count,**kwargs) if sep=='' else np.fromstring(string,dtype=dtype,count=count,sep=sep,**kwargs)


def run_env_worker(connection,config):
    env=None
    recorder=None
    trace=None
    try:
        root=Path(config.get('root',ROOT)); kind=config['benchmark']
        source=Path(config.get('source',root/'repos'/('LIBERO-plus-official' if kind=='plus' else 'LIBERO-PRO-official')))
        if kind=='clean' and 'source' not in config:
            source=Path('/data/users/jiachangzhang/DNFT4RP/upstream/libero')
        configuration=config.get('libero_config',str(root/'experiments/action_only'/f'{kind}_config'))
        if kind=='clean' and 'libero_config' not in config:
            configuration='/data/users/jiachangzhang/DNFT4RP/.libero-config'
        os.environ['LIBERO_CONFIG_PATH']=configuration
        os.environ['LD_LIBRARY_PATH']=str(root/'cache/imagemagick/root/usr/lib/x86_64-linux-gnu')+':'+str(root/'cache/osmesa/root/usr/lib/x86_64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
        os.environ['MAGICK_HOME']=str(root/'cache/imagemagick/home')
        os.environ['MAGICK_CONFIGURE_PATH']=str(root/'cache/imagemagick/root/etc/ImageMagick-6')
        os.environ['MAGICK_CODER_MODULE_PATH']=str(root/'cache/imagemagick/root/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders')
        os.environ['MAGICK_FILTER_MODULE_PATH']=str(root/'cache/imagemagick/root/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/filters')
        # Installed official init-state pickles include NumPy arrays; this process
        # loads simulator assets only, never policy checkpoints or downloaded code.
        os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
        os.environ.setdefault('MUJOCO_GL','egl'); os.environ.setdefault('PYOPENGL_PLATFORM','egl')
        sys.path.insert(0,str(source))
        from libero.libero import benchmark,get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from libero.libero.envs.bddl_utils import get_problem_info
        if kind=='plus':
            from libero.libero.envs import env_wrapper
            env_wrapper.np=LegacyBinaryNumpy()
        steps=0; done=False; success=False; current=None; instruction=''
        env_identity=None
        suites={}
        history=deque(maxlen=int(config.get('history_length',16)))
        def payload():
            key=json.dumps([spec['benchmark'],spec['suite'],spec['task_id'],spec['trial'],spec['seed'],replan_index],separators=(',',':'))
            noise_seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'little')
            return dict(observation(current,instruction),history=list(history),noise_seed=noise_seed,suite=spec["suite"],episode_key=json.dumps([spec["benchmark"],spec["suite"],spec["task_id"],spec["trial"]]))
        while True:
            message=connection.recv(); command=message.get('command')
            started=time.monotonic()
            if command=='close':
                connection.send({'ok':True}); break
            if command=='reset':
                spec=message['spec']
                if recorder is not None:
                    recorder.close(); recorder=None
                if trace is not None:
                    trace.close(); trace=None
                if spec['benchmark']!=kind:
                    raise ValueError('Benchmark mismatch; use another subprocess')
                if spec.get('manifest_path') and hashlib.sha256(Path(spec['manifest_path']).read_bytes()).hexdigest()!=spec['manifest_sha256']:
                    raise ValueError('Manifest changed since episode selection')
                random.seed(spec['seed']); np.random.seed(spec['seed'])
                if spec['suite'] not in suites:
                    suites[spec['suite']]=benchmark.get_benchmark_dict()[spec['suite']]()
                suite=suites[spec['suite']]
                task=suite.get_task(spec['task_id']); states=suite.get_task_init_states(spec['task_id'])
                if not 0<=spec['trial']<len(states):
                    raise ValueError('Initial state trial out of range')
                if spec.get('name') and task.name!=spec['name']:
                    raise ValueError('Task identity changed')
                filename=task.bddl_file.split('_view_')[0]+'.bddl' if '_view_' in task.bddl_file else task.bddl_file
                # Clean Dream-VLA evaluation follows the official task.language prompt.
                # Perturbation manifests bind the BDDL instruction so camera/light
                # filename suffixes cannot leak into the policy's instruction.
                instruction=(task.language if kind=='clean' else
                    get_problem_info(str(Path(get_libero_path('bddl_files'))/task.problem_folder/filename))['language_instruction'])
                if spec.get('language') and instruction!=spec['language']:
                    raise ValueError('Task instruction changed')
                if kind=='plus' and config.get('plus_language_mode')=='canonical_clean' and '_language_' not in task.name:
                    # Keep non-language perturbations on their original clean prompt.
                    # The Plus filename includes camera/light/layout metadata, while
                    # its BDDL can use different object/task wording from clean LIBERO.
                    import runpy
                    clean_source=Path(config.get('clean_source','/data/users/jiachangzhang/DNFT4RP/upstream/libero'))
                    task_map=runpy.run_path(str(clean_source/'libero/libero/benchmark/libero_suite_task_map.py'))['libero_task_map']
                    matches=[name for name in task_map[spec['suite']] if task.name==name or task.name.startswith(name+'_')]
                    if len(matches)!=1:
                        raise ValueError(f'Cannot uniquely map Plus task to clean prompt: {task.name}')
                    import re
                    instruction=re.sub(r'^.*SCENE\d+_','',matches[0]).replace('_',' ')
                bddl_path=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file)
                identity=(kind,spec['suite'],spec['task_id'],bddl_path)
                reused=env is not None and identity==env_identity
                if not reused:
                    if env is not None:
                        env.close(); env=None
                    env=OffScreenRenderEnv(bddl_file_name=bddl_path,camera_heights=256,camera_widths=256)
                    env_identity=identity
                env.seed(spec.get('env_seed',spec['seed']))
                env.reset(); current=env.set_init_state(states[spec['trial']])
                history.clear(); history.append(observation(current,instruction))
                for _ in range(config.get('settle_steps',30)):
                    current,_,_,_=env.step([0,0,0,0,0,0,-1])
                    history.append(observation(current,instruction))
                steps=0; replan_index=0; done=False; success=False; limit=int(config.get('max_steps_by_suite',{}).get(spec['suite'],horizon(spec['suite'])))
                if spec.get('record_dir'):
                    import imageio.v2 as imageio
                    directory=Path(spec['record_dir']); directory.mkdir(parents=True,exist_ok=True)
                    stem=f"{spec['suite']}_task{spec['task_id']}_trial{spec['trial']}"
                    recorder=imageio.get_writer(str(directory/(stem+'.mp4')),fps=20,codec='libx264',quality=7)
                    trace=(directory/(stem+'.jsonl')).open('w')
                    trace.write(json.dumps({'spec':spec,'instruction':instruction,'horizon':limit})+'\n')
                    view=observation(current,instruction)
                    recorder.append_data(np.concatenate([view['image'],view['wrist_image']],axis=1))
                connection.send({'ok':True,'observation':payload(),'done':False,'success':False,'steps':0,
                    'info':{'spec':spec,'horizon':limit,'env_reused':reused,'env_seconds':time.monotonic()-started,'settle_steps':config.get('settle_steps',30),'image_rotation_degrees':180,'gripper_convention':config.get('gripper_convention','open_01')}})
            elif command=='step':
                if env is None or done:
                    raise RuntimeError('Reset required before stepping')
                actions=convert_actions(message['action_chunk'],config.get('gripper_convention','open_01'))
                execute_steps=int(config['execute_steps'])
                if execute_steps<1 or len(actions)<execute_steps:
                    raise ValueError('Action chunk shorter than explicit execute_steps')
                if config.get('predicted_horizon') is not None and len(actions)!=int(config['predicted_horizon']):
                    raise ValueError('Action chunk differs from configured predicted_horizon')
                before=steps
                for action in actions[:execute_steps]:
                    current,_,terminal,_=env.step(action.tolist()); steps+=1
                    history.append(observation(current,instruction))
                    success=bool(terminal); done=success or steps>=limit
                    if recorder is not None:
                        view=history[-1]
                        recorder.append_data(np.concatenate([view['image'],view['wrist_image']],axis=1))
                        trace.write(json.dumps({'step':steps,'action':action.tolist(),'proprio':view['proprio'].tolist(),'success':success})+'\n')
                        if done:
                            recorder.close(); recorder=None
                            trace.close(); trace=None
                    if done: break
                replan_index+=1
                connection.send({'ok':True,'observation':payload(),'done':done,'success':success,'steps':steps-before,
                    'info':{'horizon':limit,'episode_steps':steps,'env_seconds':time.monotonic()-started}})
            else:
                raise ValueError(f'Unknown command {command}')
    except EOFError:
        pass
    except Exception as exc:
        connection.send({'ok':False,'error':repr(exc),'traceback':traceback.format_exc()})
    finally:
        if recorder is not None: recorder.close()
        if trace is not None: trace.close()
        if env is not None:
            env.close()
        connection.close()

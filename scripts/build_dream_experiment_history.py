"""Render the audited experiment ledger from saved server evidence, without training."""
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments'
def load(name):
    return json.loads((EXP/name).read_text(encoding='utf-8'))
data = load('experiment_history_20261023_evidence.json')
extra = load('experiment_history_20261023_extra.json')
final = load('experiment_history_20261023_final.json')
protocol = load('experiment_history_20261023_protocol.json')
all_data = data | extra | protocol
lines = []
sources = {}
def source(path):
    if path not in sources:
        sources[path] = f'S{len(sources)+1:02d}'
    return sources[path]
def fmt(x, places=7):
    if x is None: return '—'
    if isinstance(x, float):
        if x and (abs(x)<1e-4 or abs(x)>=1e5): return f'{x:.4g}'
        return f'{x:.{places}f}'.rstrip('0').rstrip('.')
    return str(x).replace('|','／').replace('\n','<br>')
def table(title, headers, rows, explanation):
    lines.extend([f'## {title}', '', '| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'])
    for row in rows:
        assert len(row)==len(headers),(title,row)
        lines.append('| '+' | '.join(fmt(v) for v in row)+' |')
    lines.extend(['', explanation, ''])
def cfg(run):
    v=data[run+'/resolved_config.json'];return v.get('config',v)
def med(run):return data[run+'/metrics.jsonl']['last100_median']
def wmmed(run):
    rows=extra[run+'/train/train_metrics.jsonl']['last100']
    return {k:statistics.median([r[k] for r in rows if isinstance(r.get(k),(int,float))]) for k in rows[-1] if isinstance(rows[-1][k],(int,float))}
def score(s,n):return f'{s}/{n}（{s/n*100:.2f}%）' if n else '—'

lines.extend(['# Dream-VLA baseline 复现以来的实验汇总', '',
              f'数据截点：2026-10-23 {final["snapshot"][11:19]}（服务器北京时间）；范围：2026-10-15 起的模型训练、重测和机制诊断。', '',
              '本文件按已保存配置、逐步日志、checkpoint 元数据和逐 episode 记录整理；“—”表示没有该项已核验结果，运行中的数值仅代表截点。', ''])

table('1. 名称、数据与统计口径', ['项目','本报告采用的定义'], [
['Dream baseline','官方 Dream-VLX 训练入口及模型，修正训练 bug；四套 LIBERO 混训的单模型，200K optimizer steps，并非官方四套独立训练配方的逐项复刻。'],
['E-LAF v0/v1/v2','Dream 主干 LoRA + 32 Current + 32 Delta + 8 Action tokens；共享可学习 Resampler，Delta 与 Action 双 Flow。'],
['grid11k action-only','冻结 grid encoder 11K + RoboActualizer 派生动作头；不是 E-LAF，也不是官方 V-JEPA Policy。'],
['grid11k latent WM','在上述动作头上借鉴 V-JEPA Policy 的 predictor/current-position K/V 交互；不使用 Dream 主干。'],
['grid11k small scratch','独立双视角、小型随机初始化 predictor/action expert，仍是自有实现，文本实际为 umT5。'],
['当前 Dream 双头','回到官方 Dream residual MLP 动作头；加入双视角 Current JEPA tokens 和固定绝对 Future latent Flow。'],
['encoder 11K / 103233','视觉编码器预训练步数；不能与下游 action-head、WM、E-LAF 的训练步数相加后混称。'],
['clean200 / Plus280','探索性子集：clean 40 tasks×5 trials；Plus 7 类×40 episodes，不是论文完整 benchmark。'],
['损失窗口','E-LAF：末 100 个逐步记录的逐项中位数；WM/scratch：末 100 个日志点（每 10 步一次）；各项中位数不保证相加等于 total 中位数。'],
['数据分支','Dream mixed baseline 使用官方 RLDS；E-LAF 使用其缓存；RoboActualizer 派生线路使用 1,692 条 LeRobot 示范、排除 Goal episode82，不能与 1,693 条 RLDS 混称。'],
['跨模型 loss','Action 归一化、时间采样、预测长度、precision 和 reduction 不完全相同；Delta FM、future MSE、future L1 不是同一种指标。'],
], '说明：下表同时保留模型身份和统计口径，避免把不同训练目标或不同评测协议的数值视为严格对照。')

runs=[
('E0','elaf_v0_grid11k_72token','base 新训','0→4,363','后改为 v1'),
('E1','elaf_v1_grid11k_72token','base 新训','0→19,287','失稳后暂停'),
('ER','elaf_recovery_from5000_20261019','E1 step5,000；新 optimizer','新计数0→1,000','短程恢复完成'),
('EF','elaf_fresh_flowclip_2k_20261019','base 新训','0→34,021','先2K，后续训失稳暂停'),
('EZ','elaf_lora_zero_control_20261020','EF step3,000','3,000→6,000','LoRA LR=0 对照完成'),
('V2','elaf_v2_spatial_staged_20261020_run','base 新训','0→12,742','三阶段主线；随后分叉'),
('V2-H','elaf_v2_lora10x_20261020','V2 step12,742','12,742→15,000','LoRA LR×10 分叉'),
('V2-X','elaf_v2_low_lr_from15k_20261020','V2-H step15,000','15,000→15,043','起点被用户纠正；废弃分叉'),
('V2-L1','elaf_v2_low_lr_from12742_20261020','V2 step12,742','12,742→20,000','回到低 LR 主线'),
('V2-L2','elaf_v2_low_lr_20k_to30k_20261020','V2-L1 step20,000','20,000→30,000','低 LR 续训'),
('V2-L3','elaf_v2_low_lr_to200k_20261021','V2-L2 step30,000','30,000→42,015','用户停止，未完成200K'),
]
rows=[['D-pre8','单套、8卡初试','base','未保留完整终点','停止，无 checkpoint','p4hhy9c2'],
      ['D-pre6','单套、6卡初试','base','保留5,000','早于 Flow-head DDP 修复','lz3n1xc3'],
      ['D0','Dream mixed baseline','base','0→200,000','完成','mrmlszkq']]
for tag,run,parent,steps,state in runs:
    rows.append([tag,run,parent,steps,state,data.get(run+'/run_identity.json',{}).get('wandb_run_id','—')])
for tag,run,parent,steps in [('W20','grid11k_latent_wm_20261021','action-only20K warm start','WM0→20,000'),('W57','grid11k_latent_wm_extend57k_20261021','W20完整状态','WM20,000→57,000'),('S21','grid11k_small_scratch_20261022','predictor/action 随机初始化','0→21,360')]:
    rows.append([tag,run,parent,steps,'完成',final[run]['wandb']['id']])
rows.append(['D-J','dream_jepa_dualview_20261023','Dream base 新训',f'0→{final["current_dream_dualview"]["last"]["step"]:,}','截点仍在训练','lot7ed8l'])
table('2. 实际训练与分叉总账', ['ID','实验 / run 目录','初始化 / 父实验','已执行步数区间','状态','训练 W&B ID'],rows,
      '说明：V2-H 与 V2-L1 是从同一 12,742 步分出的两条路线，W57 的 57K 则是在原 action-only20K 之外累计的 WM 更新。')

rows=[]
for tag,run,_,_,_ in runs:
    c=cfg(run);o=c['optimizer'];world=data[run+'/resolved_config.json']['world_size'];b=c['batch_size'];a=c['gradient_accumulation']
    sched=o['scheduler'];lr=f"{o['lora_lr']:g} / {o['resampler_lr']:g} / {o['new_modules_lr']:g}"
    if 'stages' in c:
        s=c['stages']['stage3'];lr=f"S1:1e-5/5e-5/1e-4；S3:{s['lora_lr']:g}/{s['resampler_lr']:g}/{s['new_modules_lr']:g}"
    if tag=='EZ':lr='0 / 5e-5 / 1e-4'
    rows.append([tag,f'{world}×{b}×{a}={world*b*a}',lr,sched,o.get('warmup_steps',str(o.get('warmup_ratio',0)*100)+'%'),
                 'FP32' if c.get('resampler_fp32') else 'BF16 autocast',o.get('clip_partition',o.get('clip_mode','global')),source(run+'/resolved_config.json')])
table('3. E-LAF 实际优化配置', ['ID','卡×micro-batch×累积=global','目标 LR：LoRA / Resampler / 新模块','调度','warmup','Resampler/Delta','clip=1 分组','来源'],rows,
      '说明：v0 的 global batch60 与后续48不同，v1 对齐 baseline LR/调度后仍失稳，而恢复与 fresh 试验又显式改变了精度和裁剪。')

table('4. 非 E-LAF 训练配置', ['ID','模型与初始化','global batch','更新','LR / 调度','Action / Future','相机 / 文本'],[
['D0','Dream base + r32/α16 all-linear LoRA，官方 MLP head',48,'200,000','1e-4；100K×0.1；无 warmup','H8 Action FM；无 Future','原生双图 + proprio；Dream 文本'],
['W20','12层 DiT-S predictor/action；复用动作头468个 tensors',240,'新增20,000','2e-4；5% warmup；cosine→2e-6','H16 Action FM + 4端点 normalized frozen-latent MSE，1:1','双相机横拼224×448；4帧间隔2；umT5'],
['W57','W20完整续训；保留模型和 optimizer',240,'再37,000，累计57K','从2e-6，500步升至5e-5，cosine→1e-5','同W20','同W20'],
['S21','约63.606M可训练参数；predictor/action随机',128,'21,360','1e-4；5% warmup；cosine→1e-6','H32 Action FM + 4端点 per-token LayerNorm L1，1:1','独立双视角224×224；4帧间隔2；实际umT5-XXL'],
['D-J','Dream base + 官方 MLP action/future heads + 输入Resampler',48,'计划200,000；运行中','1e-4；100K×0.1；无 warmup','H8 Action FM + 固定绝对Future latent FM，1:1','原生双图；JEPA双视角各16；Dream文本'],
], '说明：W20 是 warm-start 增量实验，S21 才是随机初始化下游模型的配方试验，但 S21 没有用到尚未就绪的 T5 权重。')

evals=[
('D0','dream_baseline200k_eval_corrected_20261019_143839','dream_baseline200k_eval_corrected_20261019_143839_plus_retry2',200000,4,8,'Dream'),
('G11-4','action_only/grid11k_action20k_instruction_retest_20261019_154622',None,20000,4,10,'Robo'),
('G11-10','action_only/grid11k_action20k_denoise10_20261020',None,20000,10,10,'Robo'),
('G103-4','action_only/full103233_action20k_instruction_retest_20261021',None,20000,4,10,'Robo'),
('V2-H14.5','elaf_v2_step14500_eval_gpu67_20261020',None,14500,4,8,'Dream'),
('V2-L30','elaf_v2_low_lr_step30000_eval_gpu67_20261021',None,30000,4,8,'Dream'),
('W20','grid11k_latent_wm_20261021/evaluation_020000',None,20000,4,10,'Robo'),
('W57-4','grid11k_latent_wm_extend57k_20261021/evaluation_057000',None,57000,4,10,'Robo'),
('W57-10','grid11k_latent_wm_extend57k_20261021/evaluation_057000_denoise10',None,57000,10,10,'Robo'),
('S21','grid11k_small_scratch_20261022/evaluation_021360',None,21360,4,16,'Robo'),
]
def evdata(root,bench):
    path=root+'/'+bench+'/summary.json';v=data[path]
    records=v.get('records')
    if records is None:
        records=[]
        for k,rs in extra.items():
            if k.startswith(root+'/'+bench+'/gpu_') and k.endswith('/results/episodes.jsonl'):records.extend(rs)
    assert len(records)==v['episodes'],(path,len(records),v['episodes'])
    assert sum(bool(r['success']) for r in records)==v['successes'],path
    ids=[(r.get('spec',r)['suite'],r.get('spec',r)['task_id'],r.get('spec',r).get('trial',0)) for r in records]
    assert len(set(ids))==len(ids),(path,'duplicate episodes')
    return v,records,path
rows=[];eval_records={}
for tag,clean,plus,step,denoise,execute,family in evals:
    a,ar,ap=evdata(clean,'clean');b,br,bp=evdata(plus or clean,'plus');eval_records[tag]=(ar,br)
    rows.append([tag,step,denoise,execute,score(a['successes'],a['episodes']),score(b['successes'],b['episodes']),family,source(ap)+' / '+source(bp)])
table('5. 完成的闭环评测总表', ['ID','被评 checkpoint step','去噪步数','每次执行动作数','LIBERO clean200','LIBERO-Plus280','协议族','来源'],rows,
      '说明：以上成功数已按逐 episode 记录复算且无重复，Dream 与 Robo 两种协议的差异见下一表，不能把所有行当作完全相同条件的消融。')

table('6. 闭环评测协议差异', ['项目','Dream baseline / E-LAF v2','grid action-only / latent WM','small scratch'],[
['clean horizon：Spatial/Object/Goal/Long','220 / 280 / 300 / 520','400 / 400 / 400 / 700','400 / 400 / 400 / 700'],
['settling steps',10,30,30],['Action chunk',8,16,32],['执行后重规划',8,10,16],
['默认去噪步数',4,'4；另有10步消融',4],['clean instruction','task.language','task.language','task.language'],
['Plus instruction','canonical_clean：非语言扰动用对应clean指令，语言类保留改写（Dream与v2同一launcher分支）','保留manifest BDDL原始/改写指令','保留manifest BDDL原始/改写指令'],
['动作/状态归一化','Dream官方bounds_q99；对应suite统计','RoboActualizer min/max；继承grid11k stats','继承grid11k min/max stats'],
['评测子集','clean200 / Plus280','clean200 / Plus280','clean200 / Plus280'],
], '说明：特别是 Plus 非语言类的指令替换、settling 和 horizon 不同，会影响跨协议比较的解释。')

def group_score(records,field,name):
    selected=[r for r in records if r.get('spec',r).get(field)==name]
    return score(sum(bool(r['success']) for r in selected),len(selected))
suites=['libero_spatial','libero_object','libero_goal','libero_10']
table('7. clean200 分 suite 结果', ['ID','Spatial','Object','Goal','LIBERO-10'],[
    [tag]+[group_score(eval_records[tag][0],'suite',s) for s in suites] for tag,*_ in evals],
    '说明：每个 suite 固定50个 episodes，14.5K为高LR分支、30K为低LR分支，不能解释为同一模型的连续提升曲线。')
cats=['Camera Viewpoints','Robot Initial States','Language Instructions','Light Conditions','Background Textures','Sensor Noise','Objects Layout']
table('8. Plus280 分扰动类别结果', ['ID','视角','初始状态','语言','光照','背景','传感噪声','布局'],[
    [tag]+[group_score(eval_records[tag][1],'category',c) for c in cats] for tag,*_ in evals],
    '说明：每类40个 episodes，单次成功对应2.5个百分点，类别结果均由原始记录重算。')
table('9. Plus280 分 suite 结果', ['ID','Spatial','Object','Goal','LIBERO-10'],[
    [tag]+[group_score(eval_records[tag][1],'suite',s) for s in suites] for tag,*_ in evals],
    '说明：Plus 的 suite 统计与上一表使用同一批280条记录，仅改变分组维度。')

table('10. 早期无效评测与协议定位', ['实验 / 干预','结果','样本','效力 / 问题','来源'],[
['baseline200K 最初错误协议','26/200 = 13.00%','clean200','作废：实际1步而非4步；还叠加语言、模拟器、settling/horizon差异',source('dream_baseline200k_eval_20261019_132600/clean/summary.json')],
['仅切4步的早期重跑','不纳入正式成绩','中间重跑','仍有BDDL语言与模拟器不一致；被corrected取代',source('dream_baseline200k_eval_flow4_20261019_134959/invalid_protocol.json')],
['原生官方 Goal 控制','10/10','10 tasks×1初态','小样本接口控制',source('dream_eval_audit_20261019/findings.json')],
['共享评测器+官方模拟器+旧语言','8/10','同Goal控制','语言尚未修正','同上'],
['共享评测器+旧模拟器+旧语言','3/10','同Goal控制','模拟器依赖差异明显','同上'],
['共享评测器+官方模拟器+canonical语言','10/10','同Goal控制','修正语言后的控制','同上'],
['最终生产RPC控制','9/10','同Goal控制','并行数值/轨迹敏感；非完整成绩','同上'],
['单套5K checkpoint / tiny模型 / base Plus smoke','仅接口验证','少量episodes','不可当作修正baseline或成熟模型的质量成绩','DREAM_VLA_OFFICIAL_TRAINING.md'],
], '说明：13%已被明确标记为无效协议结果，正式修正后的 Dream baseline 是195/200，而不是模型训练本身仅有13%成功率。')

rows=[]
for tag,run,*_ in runs:
    m=med(run);last=data[run+'/metrics.jsonl']['last']['step']
    rows.append([tag,last,m.get('train/action_loss'),m.get('train/delta_loss'),m.get('train/loss'),m.get('train/grad_preclip/lora'),m.get('train/seconds_per_update'),source(run+'/metrics.jsonl')])
table('11. E-LAF 各 run 终点训练统计', ['ID','终点step','Action loss','Delta loss','Total loss','LoRA preclip范数','秒/update','来源'],rows,
    '说明：这里均为该run末100个记录的逐项中位数，V2-X仅运行43步，因此使用其全部43个记录。')

rows=[['D0','195,001–200,000','跨6卡、30,000个microbatch均值',final['baseline_checkpoint']['selection_loss'],'不适用',1e-5,source('baseline_checkpoint（experiment_history_20261023_final.json）')],
['D0','W&B step199,990','rank0最后一次日志单点',final['baseline_wandb_summary']['VLA Train/Loss'],'不适用',1e-5,source('baseline_wandb_summary（experiment_history_20261023_final.json）')]]
for tag,run in [('W20','grid11k_latent_wm_20261021'),('W57','grid11k_latent_wm_extend57k_20261021'),('S21','grid11k_small_scratch_20261022')]:
    m=wmmed(run);rs=extra[run+'/train/train_metrics.jsonl']['last100']
    rows.append([tag,f"{rs[0]['step']}–{rs[-1]['step']}",'末100个日志点中位数',m['train/action_loss'],m['train/image_loss'],m['train/lr'],source(run+'/train/train_metrics.jsonl')])
table('12. Baseline、WM、scratch 的终段损失', ['ID','窗口','聚合口径','Action loss','Future loss','LR','来源'],rows,
    '说明：baseline的0.00543单点不能替代0.02116跨卡窗口均值，WM/scratch的更低Action数值也不能直接解释成更高成功率。')

rows=[]
for tag,run in [('E1','elaf_v1_grid11k_72token'),('EF','elaf_fresh_flowclip_2k_20261019')]:
    ws=data[run+'/metrics.jsonl']['windows']
    keys=['0-500','500-1000','1000-1500','1500-2000','2000-3000'] if tag=='E1' else ['2000-3000','3000-3500','3500-4000','4000-4500','4500-6000','5800-5850','5850-5900','6000-7000','33000-35000']
    for key in keys:
        if key not in ws:continue
        m=ws[key]['median'];rows.append([tag,key,ws[key]['n'],m.get('train/action_loss'),m.get('train/delta_loss'),m.get('train/grad/global_preclip'),m.get('train/grad_preclip/lora'),m.get('train/update_relative/lora')])
table('13. 两轮失稳的时间窗口', ['ID','step区间[左闭右开]','记录数','Action','Delta','global preclip','LoRA preclip','LoRA相对更新量'],rows,
    '说明：fresh轮在3K后LoRA梯度先增长，约5,850步Action突变，后期巨梯度不能等同于每一步参数更新同比例变大。')

table('14. 5000起点恢复与固定batch优化检查', ['检查','样本 / 条件','Action','Delta模型MSE','解析MSE','零速度MSE','其他'],[
['退化step5000恢复1000步','新optimizer；200warmup；FP32；分组clip',med('elaf_recovery_from5000_20261019')['train/action_loss'],med('elaf_recovery_from5000_20261019')['train/delta_loss'],'—','—','不是失稳前恢复'],
['恢复终点解析检查','固定4样本；u=s=.5','—',.985690,.001259791,.998078,'Delta RMS=.0177468'],
['全新固定batch：update0','4样本；固定噪声；无aug',1.816097,1.661134,.017317,'—','新LoRA/Resampler/heads'],
['全新固定batch：update60','同4样本与噪声',.012743,.393323,.015278,'—','仅证明可优化/记忆小batch'],
], '说明：恢复模型没有学好简单去噪映射，而新初始化小batch可以降低两项损失，但这不是动力学泛化证据。')

rows=[]
for step in [0,1000,1500,2000]:
    p=f'elaf_fresh_flowclip_2k_20261019/slot_diagnostics/metrics-{step:06d}.json';m=extra[p]['metrics']
    q=f'elaf_fresh_flowclip_2k_20261019/diagnostics/diagnostics-step-{step:09d}.json';d=extra[q]['metrics']
    rows.append([step,f"{m['monitor/z/unique_slot_energy']*100:.5f}%",f"{m['monitor/delta/unique_slot_energy']*100:.5f}%",m['monitor/attention/current/entropy_normalized'],m['monitor/attention/current/per_head_pairwise_jsd_nats'],d['monitor/z/slot_cka_mean'],d['monitor/z/global_erank'],d['monitor/delta/rms_raw']])
table('15. 随机Query Resampler的冗余诊断', ['step','Current unique能量','Delta unique能量','attention熵/max','per-head JSD(nats)','CKA均值','Current有效秩','Delta RMS'],rows,
    '说明：固定1024对样本显示slot开始分化但共同变化仍占主导，这些attention与能量指标比仅凭CKA更直接。')

table('16. Step2000 slot-specific消融', ['Action时间s','原始Action MSE','移除Current/Delta特异变化后','相对变化'],[
[.25,.13776,.13813,'+0.27%'],[.5,.21841,.21948,'+0.49%'],[.75,.31959,.31951,'−0.03%']],
    '说明：32个固定样本、保留slot固定偏置与共同成分的离线消融表明，当时动作预测对slot-specific变化不敏感，而不是整个Future分支无用。')
table('17. Step2000 Future信息路径消融', ['干预','Action velocity MSE','相对正常'],[
['正常Joint Flow',.214463,'参考'],['打乱干净Delta，保留各样本噪声',.215986,'+0.71%'],['逐层禁止Action读取Delta',.217412,'+1.38%'],['逐层禁止Action读取Current JEPA',.564316,'+163.13%'],['逐层禁止所有非Delta token读取Delta',.228306,'+6.45%']],
    '说明：固定32样本、s=.5及10个u中点、原生FlashAttention的最终有效诊断支持Current信息使用更明显，但不是闭环成功率消融。')
model=[.073867,.068261,.064092,.061521,.062264,.065977,.077050,.104231,.185789,.548756]
analytic=[.002075,.002592,.003329,.004432,.006190,.009247,.015286,.029962,.083226,.749038]
table('18. Step2000 Delta按时间的解析基线', ['u','模型Delta MSE','−d_u/(1−u) MSE','零速度MSE'],[
    [round(.05+i*.1,2),model[i],analytic[i],1.003167] for i in range(10)],
    '说明：九个中点上模型不及无参数解析去噪器，接近u=1时解析式自身放大误差，不能仅凭最后一行宣称学到了动力学。')

table('19. 健康/失稳checkpoint逐层诊断及LoRA零LR对照', ['项目','EF step3000','EF step34021','EZ：3000→6000'],[
['固定batch Action MSE',.1751,2.1502,'终段训练中位数 '+fmt(med('elaf_lora_zero_control_20261020')['train/action_loss'])],
['固定batch Delta MSE',.03939,1.9142,'终段训练中位数 '+fmt(med('elaf_lora_zero_control_20261020')['train/delta_loss'])],
['Action→LoRA梯度范数',5.295,239.976,'LoRA LR=0，保留requires_grad'],
['Delta→LoRA梯度范数',3.136,67871895,'其他组维持原配置'],
['两任务LoRA gradient cosine',-.00233,-.03435,'不重置非LoRA状态'],
['block27 Q LoRA/base RMS',.890,45.383,'完整对照已完成'],
['block27 K LoRA/base RMS','—',26.204,'LoRA权重逐项hash不变：True'],
], '说明：健康/失稳列为固定4样本×3时间点的诊断中位数，零LR列为训练终段统计，两者不同口径，但冻结对照确实稳定完成到6K。')

init=data['elaf_v2_spatial_staged_20261020_run/ELAF_V2_INIT_DIAGNOSTICS.json'];old=init['random_query_init']['metrics'];new=init['spatial_anchored_init']['metrics']
fields=[('Current unique energy','monitor/z/unique_slot_energy'),('Delta unique energy','monitor/delta/unique_slot_energy'),('Current CKA均值','monitor/z/slot_cka_mean'),('Current有效秩','monitor/z/global_erank'),('Delta有效秩','monitor/delta/global_erank'),('Delta RMS','monitor/delta/rms_raw'),('per-head Query JSD','monitor/attention/current/per_head_pairwise_jsd_nats'),('attention熵/max','monitor/attention/current/entropy_normalized')]
table('20. v2空间锚定初始化对照', ['指标','原随机Query初始化','空间锚定初始化'],[[label,old.get(key),new.get(key)] for label,key in fields],
    '说明：同一1024样本、FP32重测，真实网格2×24×24池化为2×4×4，Z=P+0.1×attention残差保留空间结构，unique energy此表以0–1比例表示。')
table('21. v2三阶段实际执行', ['阶段','step边界','LoRA','Resampler LR','新模块LR','状态迁移证据'],[
['S1','0–2999','1e-5；200步warmup',5e-5,1e-4,'base新初始化；A随机/B零'],
['S2','3000–9999','requires_grad=False；权重冻结',5e-5,1e-4,'step3000与10000 LoRA hash完全相同'],
['S3低LR','10000起','1000步warmup→2e-6',2e-5,5e-5,'重新创建LoRA moments；保留非LoRA状态；cosine最低比0.1'],
['S3高LR分叉','12742→15000','目标2e-5',2e-5,5e-5,'独立V2-H；14500评测来自此分支'],
], '说明：三阶段切换已有transitions记录，低LR主线从12,742重新分叉续到42,015，不经过高LR15K权重。')
rows=[]
for tag,run,steps in [('V2','elaf_v2_spatial_staged_20261020_run',[0,3000,10000,12000]),('V2-H','elaf_v2_lora10x_20261020',[15000]),('V2-L1','elaf_v2_low_lr_from12742_20261020',[20000]),('V2-L2','elaf_v2_low_lr_20k_to30k_20261020',[30000]),('V2-L3','elaf_v2_low_lr_to200k_20261021',[42000])]:
    for step in steps:
        path=f'{run}/v2_diagnostics/representation-{step:09d}.json'
        if path not in extra:continue
        m=extra[path]['metrics'];rows.append([tag,step,m.get('monitor/delta/rms_raw'),m.get('monitor/z/global_erank'),m.get('monitor/delta/global_erank'),m.get('monitor/z/slot_cka_mean'),m.get('monitor/z/unique_slot_energy'),m.get('monitor/delta/unique_slot_energy')])
table('22. v2阶段性表征监控', ['ID','step','Delta RMS','Current有效秩','Delta有效秩','CKA均值','Current unique','Delta unique'],rows,
    '说明：unique为0–1比例，空间锚定后的多样性明显不同于随机Query旧版，但这些指标不替代闭环控制结果。')

rows=[]
for tag,run,intervals in [('V2','elaf_v2_spatial_staged_20261020_run',['2000-3000','10000-11000','11000-12000','12000-12743']),('V2-H','elaf_v2_lora10x_20261020',['14000-15000']),('V2-L1','elaf_v2_low_lr_from12742_20261020',['14000-15000','19000-20000']),('V2-L2','elaf_v2_low_lr_20k_to30k_20261020',['20000-21000','29000-30000'])]:
    for interval in intervals:
        w=data[run+'/metrics.jsonl']['windows'].get(interval)
        if w:
            m=w['median'];rows.append([tag,interval,w['n'],m.get('train/action_loss'),m.get('train/delta_loss'),m.get('train/grad_preclip/lora'),m.get('train/lr/lora')])
table('22a. v2学习率分叉与下降速度的窗口对照', ['ID','step区间[左闭右开]','记录数','Action中位数','Delta中位数','LoRA preclip','LoRA LR'],rows,
    '说明：这些固定窗口可用于比较10K–20K和20K–30K的下降趋势，但中途分叉与学习率变化需按ID区分。')

rows=[]
for tag,run in [('W20','grid11k_latent_wm_20261021'),('W57','grid11k_latent_wm_extend57k_20261021'),('S21','grid11k_small_scratch_20261022')]:
    m=wmmed(run);rows.append([tag,m.get('train/action_loss'),m.get('train/image_loss'),m.get('train/world_prediction_mse',m.get('train/image_loss')),m.get('train/world_zero_mse'),m.get('train/world_copy_current_mse'),m.get('train/grad_norm'),m.get('train/action_sigma_mean')])
table('23. Latent WM与scratch的预测指标', ['ID','Action MSE','Future训练loss','Future预测MSE','零预测MSE','复制当前MSE','grad norm','Action sigma均值'],rows,
    '说明：W20/W57的Future训练loss是MSE，S21是L1而另记预测MSE，低训练loss并未保证额外训练后成功率单调提升。')
rows=[]
precision=data['grid11k_small_scratch_20261022/precision_audit.json']
for a,b in zip(precision['fp32'],precision['bf16']):rows.append([a['sigma'],a['mse'],b['mse'],a['future_l1'],b['future_l1'],a['finite'] and b['finite']])
table('24. Scratch数值精度检查', ['sigma','FP32 Action MSE','BF16 Action MSE','FP32 Future L1','BF16 Future L1','两者finite'],rows,
    f"说明：固定检查中动作差异RMS={precision['action_difference']['rms']:.6f}、max={precision['action_difference']['max']:.6f}，未见足以据此解释成功率差距的精度灾难。")

rows=[]
seen=set()
for path in ['libero_encoder_latent_comparison_20261021/summary.json','libero_encoder_longrun_latents_20261021/summary.json']:
    for name,m in data[path]['medians'].items():
        if name in seen:continue
        seen.add(name);rows.append([name,m['rms'],m['delta_rms'],m['relative_delta'],m['spatial_erank'],m['spatial_mean_cosine'],m['temporal_cosine'],m['pixel_delta_spearman']])
table('24a. Encoder 11K之后的整体latent对照', ['encoder','latent RMS','Delta RMS','相对Delta','空间有效秩','patch平均cosine','时序cosine','像素/latent变化Spearman'],rows,
    '说明：同16个episode的32对示范窗口，grid11K→17,358是同一短训线，full10K→100K是相同配方的另一独立run，不能称100K为grid11K直接续训。')

region=extra['libero_encoder_final103233_regions_20261021/task_regions/summary.json']
models=['official','grid11000','full10000','full40000','full100000','full103233']
table('25. Encoder任务区域跨时刻检索', ['encoder','操作物体Top-K命中','目标区域Top-K命中','夹爪Top-K命中','操作物体AUC','操作物体raw AUC'],[
    [m]+[f"{region[c][m]['future_topk_recall']*100:.2f}%" for c in ['object','target','gripper']]+[region['object'][m]['future_auc'],region['object'][m]['raw_future_auc']] for m in models],
    '说明：只有三个手工近似ROI案例、以当前区域原型检索t+8未来区域，K等于目标框patch数，不能将局部对应改善直接等同于控制成功率改善。')
sep=extra['libero_encoder_final103233_regions_20261021/region_separation/summary.json']
rows=[]
for c in ['object','target','gripper']:
    for m,v in sep[c].items():rows.append([c,m,v['within_cosine'],v['background_margin'],v['distractor_margin']])
table('26. Encoder区域内部一致性与可分性', ['区域','encoder','区域内cosine','相对背景margin','相对干扰物margin'],rows,
    '说明：这组局部latent统计与G103下游重测51%/30%并存，说明“看得到关键物体”和“动作头能有效使用”是不同问题。')

table('27. 当前Dream双头实验的最终实际配置', ['项目','配置 / 继承来源'],[
['起点','Dream-VLA base；不加载旧E-LAF或mixed baseline训练权重'],
['主干','实际hidden3584、28层；LoRA r32/alpha16 all-linear；BF16'],
['原生条件','保留Dream图像、语言、proprio输入及原生图像增强'],
['JEPA当前输入','冻结grid11k；主视角16+wrist16；shared spatial-anchored Resampler；各自4帧'],
['真实网格 / tubelet','384分辨率，Conv3d tubelet2×patch16×16；2×24×24→1×4×4'],
['空间/相机编码','新增固定2D sin/cos + learned camera/type；主干原RoPE保留'],
['Future目标','主视角固定池化16×1024绝对latent；不经过可学习目标Resampler；t+8；无Delta'],
['输出头','Action保留官方residual MLP；Future沿用相同MLP模板、输出1024'],
['交互','Action/Future tokens在Dream各层交互，非独立两套专家Transformer'],
['20% Future dropout','按样本屏蔽所有非Future query读取Future keys（含Future时间），保留Future监督'],
['训练 / 推理','Action/Future FM=1:1；独立训练时间；4-step联合Euler推理'],
['数值 / batch','BF16主干；FP32 Resampler/target/loss；6×8=48；gradient checkpointing关闭'],
['优化器 / 保存','官方AdamW LR1e-4、100K×0.1、无warmup；每5K保存完整latest/best'],
['当前step / 终段','step '+str(final['current_dream_dualview']['last']['step'])+'；'+', '.join(f'{k}={statistics.median(r[k] for r in final["current_dream_dualview"]["last100"]):.5f}' for k in ['action_loss','future_loss','seconds_per_update'])],
['闭环成功率','尚未评测'],
], '说明：这是用户重新确认的官方MLP双头方向，已与旧72-token Delta-Flow E-LAF分开记录，当前仅有早期训练数据。')

table('28. 已准备但没有正式质量结果的工作', ['项目','已执行','尚缺 / 不应宣称'],[
['V-JEPA Policy官方代码','保存revision d0e75c44463f85942f3580813fbcfe904cc0cf15；imports/CUDA/server CLI通过','所存状态记录官方policy checkpoint及匹配stats未就绪，未完成本机官方权重rollout'],
['T5下载','曾使用代理、镜像及续传；S21按授权改用umT5','不能把S21标为T5-XXL实验'],
['LIBERO-Pro / RoboTwin / RoboDojo','接口/依赖与部分headless smoke准备','没有本报告模型的完整有效成功率可列'],
['infra缓存改动','已开发部分缓存/读取改动并做独立测试','按最新要求暂不继续；不计入模型实验结果'],
], '说明：下载、环境和smoke不等于模型质量实验，公开论文中的97.25%/98.70%等也不作为我们自己的结果入表。')

# Preserve provenance without copying secrets, credentials, full commands, or tensor payloads.
for p in ['elaf_fresh_flowclip_2k_20261019/slot_ablation/results.json','elaf_fresh_flowclip_2k_20261019/branch_usage_native/results.json','elaf_layer_diagnostic_20261020/results.json','elaf_lora_zero_control_20261020/completion_verification.json','elaf_v2_spatial_staged_20261020_run/ELAF_V2_INIT_DIAGNOSTICS.json','elaf_v2_spatial_staged_20261020_run/transitions.jsonl','grid11k_small_scratch_20261022/precision_audit.json','libero_encoder_final103233_regions_20261021/task_regions/summary.json','libero_encoder_final103233_regions_20261021/region_separation/summary.json']:
    source(p)
table('29. 原始证据索引', ['编号','相对服务器experiments目录的路径 / 本地证据键'],[[v,k] for k,v in sources.items()],
    '说明：服务器根目录为 /data/users/jiachangzhang/eb_vjepa/experiments，本地同目录的 experiment_history_20261023_{evidence,extra,final,protocol}.json 保存了本次整理所用快照。')

output=EXP/'DREAM_VLA_EXPERIMENT_HISTORY_20261023.md'
output.write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(output=str(output),lines=len(lines),tables=sum(s.startswith('## ') for s in lines),evaluations=len(evals),source_entries=len(sources)),ensure_ascii=False))

"""Native simulator loops, no model instantiation inside the simulator."""
def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name='reset')
    while not TASK_ENV.is_episode_end():
        model_client.call(func_name='update_obs', obs=TASK_ENV.get_obs())
        actions = model_client.call(func_name='get_action')
        for action in actions:
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end():
                break


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name='reset')
    while not TASK_ENV.is_episode_end():
        ids = TASK_ENV.get_running_env_idx_list()
        if not ids:
            break
        obs = TASK_ENV.get_obs_batch(ids)
        # Native observations may omit env_idx; preserve the simulator's actual ordering.
        obs = [dict(item, env_idx=int(idx)) for idx, item in zip(ids, obs)]
        model_client.call(func_name='update_obs_batch', obs=obs)
        chunks = model_client.call(func_name='get_action_batch', obs=ids)
        if len(chunks) != len(ids) or not chunks or not chunks[0]:
            raise ValueError('Invalid action batch from E-LAF adapter')
        horizon = len(chunks[0])
        if any(len(chunk) != horizon for chunk in chunks):
            raise ValueError('Unequal action horizons')
        for step in range(horizon):
            running = set(TASK_ENV.get_running_env_idx_list())
            active = [(idx, chunk) for idx, chunk in zip(ids, chunks) if idx in running]
            if not active:
                break
            TASK_ENV.take_action_batch([chunk[step] for _, chunk in active], [idx for idx, _ in active])
            if TASK_ENV.is_episode_end():
                break

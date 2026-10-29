"""Joint Euler sampling with official shifted action-token readout."""
import torch


@torch.inference_mode()
def sample_joint_flow(model, action_head, noisy_action_projector, model_inputs,
                      action_mask, primary_features, wrist_features, steps=4,
                      generator=None):
    if steps < 1:
        raise ValueError('steps must be positive')
    b = primary_features.shape[0]
    device = primary_features.device
    if not torch.all(action_mask.sum(1) == 56):
        raise ValueError('Official LIBERO action layout requires 56 scalar tokens')
    action = torch.randn((b,8,7),device=device,dtype=torch.float32,generator=generator)
    future = torch.randn((b,16,1024),device=device,dtype=torch.float32,generator=generator)
    inputs = dict(model_inputs)
    inputs.pop('labels',None)
    inputs.update(all_actions_mask=action_mask, jepa_primary=primary_features,
                  jepa_wrist=wrist_features, future_dropped=torch.zeros(b,device=device,dtype=torch.bool),
                  use_cache=False, return_dict=True, num_logits_to_keep=1)
    for i in range(steps):
        time = torch.full((b,),i/steps,device=device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output = model(**inputs,
                noisy_action_features=noisy_action_projector(action.to(torch.bfloat16).reshape(b,56,1)),
                timestep_embeddings=action_head.time_encoder(time.to(torch.bfloat16)).unsqueeze(1).to(torch.bfloat16),
                noisy_future=future, future_time=time)
            hidden = output.hidden_states[:, :-1][action_mask[:,1:]].reshape(b,56,-1)
            velocity = action_head.predict_velocity(hidden)
        action = action + velocity.float()/steps
        future = future + output.future_velocity.float()/steps
        # Native image embedding cache; includes no future/JEPA tokens.
        inputs['inputs_embeds'] = output.inputs_embeds
        inputs.pop('pixel_values',None)
        inputs.pop('image_grid_thw',None)
    return action, future

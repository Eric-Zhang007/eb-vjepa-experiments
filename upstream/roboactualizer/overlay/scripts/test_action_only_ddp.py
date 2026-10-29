"""Two CPU/Gloo ranks exercising action-only DDP and no_sync; unit fixtures only."""
from contextlib import nullcontext
import json

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from test_action_only import make_model


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        torch.manual_seed(239)
        model = DistributedDataParallel(make_model(), find_unused_parameters=True,
                                        gradient_as_bucket_view=True)
        optimizer = torch.optim.AdamW(model.module.trainable_parameters(), lr=1e-3)
        torch.manual_seed(239 + rank)
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            for microstep in range(2):
                with model.no_sync() if microstep == 0 else nullcontext():
                    batch = dict(z0=torch.randn(2, 4, 16), text_context=torch.randn(2, 5, 8),
                        text_mask=torch.ones(2, 5, dtype=torch.bool), proprio=torch.randn(2, 3),
                        action_chunk=torch.randn(2, 3, 2), action_is_pad=torch.zeros(2, 3, dtype=torch.bool))
                    loss, _ = model(batch)
                    assert torch.isfinite(loss)
                    (loss / 2).backward()
            used, unused = [], []
            for name, parameter in model.module.named_parameters():
                if not parameter.requires_grad:
                    assert parameter.grad is None
                    continue
                if parameter.grad is None:
                    unused.append(name)
                    continue
                mean = parameter.grad.detach().clone()
                dist.all_reduce(mean)
                mean /= dist.get_world_size()
                torch.testing.assert_close(parameter.grad, mean)
                used.append(name)
            assert used
            optimizer.step()
        if rank == 0:
            print(json.dumps({"ddp_unit": "PASS", "world_size": dist.get_world_size(),
                "steps": 3, "accumulation": 2, "used_parameters": len(used),
                "unused_parameters": unused, "real_training_acceptance": False}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

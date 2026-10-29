"""Eight-rank tiny-model regression: accumulation, EMA, resume, W&B offline.

Not a V-JEPA training run or throughput benchmark.
"""
import contextlib
import copy
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from model import ema_update
from runtime import atomic_save, restore_rng, rng_state, StepSchedule
from train import optimizer_for, wandb_start


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 8)
        self.predictor = torch.nn.Linear(8, 4)
        self.unused_image_branch = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.predictor(torch.nn.functional.gelu(self.encoder(x))).square().mean()


def update(ddp, optimizer, schedule, target, step):
    optimizer.zero_grad(set_to_none=True)
    schedule.advance(step)
    for micro in range(2):
        with contextlib.nullcontext() if micro else ddp.no_sync():
            x = torch.randn(2, 4, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = ddp(x)
            (loss / 2).backward()
    gradient = ddp.module.encoder.weight.grad
    gathered = [torch.empty_like(gradient) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, gradient)
    for other in gathered:
        torch.testing.assert_close(other, gradient, rtol=0, atol=0)
    assert ddp.module.unused_image_branch.weight.grad is None
    optimizer.step()
    ema_update(ddp.module.encoder, target, 0.99925)
    assert all(p.grad is None for p in target.parameters())


def main():
    root = Path("/data/users/jiachangzhang/eb_vjepa")
    output = root / "experiments/ctp_v1/unit_ddp"
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    torch.manual_seed(239 + rank)
    dist.init_process_group("nccl")
    try:
        config = yaml.safe_load(Path(__file__).with_name("config.yaml").read_text())
        net = Tiny().cuda()
        ddp = DDP(net, device_ids=[local], find_unused_parameters=True, gradient_as_bucket_view=True)
        target = copy.deepcopy(net.encoder).requires_grad_(False)
        optimizer = optimizer_for(net, config)
        schedule = StepSchedule(optimizer, config)
        update(ddp, optimizer, schedule, target, 1)
        states = [None] * 8 if rank == 0 else None
        dist.gather_object(rng_state(), states, dst=0)
        if rank == 0:
            atomic_save({
                "model": net.state_dict(), "target": target.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": schedule.state_dict(),
                "rng": states,
            }, output / "checkpoint.pt")
        dist.barrier()
        update(ddp, optimizer, schedule, target, 2)
        expected = copy.deepcopy(net.state_dict())
        expected_target = copy.deepcopy(target.state_dict())
        del ddp, optimizer, net
        checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=False)
        net = Tiny().cuda()
        net.load_state_dict(checkpoint["model"], strict=True)
        target = copy.deepcopy(net.encoder).requires_grad_(False)
        target.load_state_dict(checkpoint["target"], strict=True)
        optimizer = optimizer_for(net, config)
        optimizer.load_state_dict(checkpoint["optimizer"])
        schedule = StepSchedule(optimizer, config)
        schedule.load_state_dict(checkpoint["scheduler"])
        ddp = DDP(net, device_ids=[local], find_unused_parameters=True, gradient_as_bucket_view=True)
        restore_rng(checkpoint["rng"][rank])
        update(ddp, optimizer, schedule, target, 2)
        for name, value in net.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        for name, value in target.state_dict().items():
            torch.testing.assert_close(value, expected_target[name], rtol=0, atol=0)
        if rank == 0:
            config["name"] = "ctp_v1_tiny_ddp_unit_test"
            config["wandb_mode"] = "offline"
            run = wandb_start(config, "unit", {"test_only": True}, None)
            run.log({"unit/exact_resume": 1, "unit/world_size": 8}, step=1)
            run.finish()
            report = {"passed": True, "world_size": 8, "exact_resume": True,
                      "no_sync_accumulation": True, "wandb_offline_log": True,
                      "pretrained_smoke": False}
            (output / "result.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

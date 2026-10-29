import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    dev = rank % torch.cuda.device_count()
    torch.cuda.set_device(dev)
    t = torch.ones(4, device="cuda")
    dist.all_reduce(t)
    ok = bool((t == world).all().item())
    print("NCCL rank %d/%d dev %d allreduce_ok=%s" % (rank, world, dev, ok), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

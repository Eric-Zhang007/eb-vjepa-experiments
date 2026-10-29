"""CPU checks for outcome proportions, rank coverage and exact replay."""
import json
from pathlib import Path
import torch
from runtime import OutcomeBatches, StepSchedule


def check_sampler():
    records = [{"outcome": int(i < 673)} for i in range(1000)]
    for accumulation in (1, 2, 4):
        batch = 16 // accumulation
        def stream(rank, start=0):
            return list(OutcomeBatches(records, batch, rank, 8, 239, start, 200 * accumulation,
                                       accumulation, 673, 1000))
        ranks = [stream(rank) for rank in range(8)]
        successes = 0
        for step in range(200):
            gathered = [item for sub in range(accumulation) for rank in range(8)
                        for item in ranks[rank][step * accumulation + sub]]
            assert len(gathered) == 128
            n = sum(records[index]["outcome"] for index, _ in gathered)
            assert n in (86, 87), n
            successes += n
            assert abs(successes - (step + 1) * 128 * .673) <= .5 + 1e-8
        for rank in range(8):
            assert stream(rank, 37 * accumulation + 1) == ranks[rank][37 * accumulation + 1:]
    print("PASS: 8-rank global proportions, accumulation, rounding and resume")


def check_schedule():
    parameter = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.AdamW([parameter], lr=2e-6)
    config = dict(warmup_steps=2, max_steps=1000000, start_lr=2e-6, lr=2e-5,
                  final_lr=2e-6, lr_clock="wall", wall_seconds=100)
    schedule = StepSchedule(opt, config)
    assert schedule.advance(1, 5) > 2e-6
    assert abs(schedule.advance(2, 10) - 2e-5) < 1e-12
    saved = schedule.state_dict()
    expected = schedule.advance(3, 55)
    resumed = StepSchedule(opt, config)
    resumed.load_state_dict(saved)
    assert resumed.advance(3, 55) == expected
    assert abs(expected - 1.1e-5) < 1e-12
    assert abs(resumed.advance(4, 100) - 2e-6) < 1e-12
    print("PASS: wall-clock cosine endpoints and resume")


if __name__ == "__main__":
    check_sampler()
    check_schedule()
    report = Path("/data/users/jiachangzhang/eb_vjepa/experiments/ctp_full_72h/data_report.json")
    if report.exists():
        data = json.loads(report.read_text())
        assert data["downloaded_videos"] == 813
        assert data["videos"] == data["metadata_referenced_videos"] == 784
        assert len(data["cameras"]) == 3 and all(data["cameras"].values())
        assert sum(data["full_episode_counts"].values()) == 95658
        print("PASS: real full DROID manifest")

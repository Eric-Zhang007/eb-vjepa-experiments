"""Compare the dataset decoder against sequential reference AV1 RGB frames."""
import argparse
import av
import torch
from Roboactualizer.datasets.lerobot.lerobot.datasets.video_utils import decode_video_frames_pyav

parser = argparse.ArgumentParser()
parser.add_argument('video')
args = parser.parse_args()
frames, times = [], []
with av.open(args.video) as container:
    for i, frame in enumerate(container.decode(video=0)):
        frames.append(torch.from_numpy(frame.to_ndarray(format='rgb24')).permute(2, 0, 1))
        times.append(float(frame.time))
        if i == 20:
            break
indices = [6, 0, 6, 2, 20]
actual = decode_video_frames_pyav(args.video, [times[i] for i in indices], 1e-4)
expected = torch.stack([frames[i] for i in indices]).float() / 255
torch.testing.assert_close(actual, expected, atol=0, rtol=0)
try:
    decode_video_frames_pyav(args.video, [-1], 1e-4)
except ValueError:
    pass
else:
    raise AssertionError('Negative timestamp accepted')
print('PASS native AV1 decoder: exact RGB, duplicate/unsorted queries, timestamp validation')

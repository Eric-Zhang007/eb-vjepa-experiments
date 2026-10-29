# Dream-VLA checkpoint download

Started 2026-10-15 on Autolab as a detached, resumable background process.
The launch was verified by `/proc/958288/cmdline` and initial shard bytes.
The PID is historical evidence only; inspect its command line before acting.

- Model: `Dream-org/Dream-VLA-7B`
- Pinned revision: `0cd0e25c1771c5b468c74585b5cf35c4a4aa315c`
- Transport: `https://hf-mirror.com`, no proxy needed at launch.
- Destination: `/data/users/jiachangzhang/eb_vjepa/checkpoints/dream_vla/Dream-VLA-7B`
- Log: `/data/users/jiachangzhang/eb_vjepa/experiments/elaf/download_dream_vla.log`
- PID file: `/data/users/jiachangzhang/eb_vjepa/experiments/elaf/download_dream_vla.pid`
- Downloader: `scripts/download_dream_vla.py` (local and remote).

All 19 non-weight files, including README, configuration, model implementation,
tokenizer assets and shard index, were downloaded and verified at launch.
Four safetensors shards total 16,634,226,144 bytes. They are still downloading;
the checkpoint must not be treated as usable until the completion receipt exists.

Each file is checked against pinned API metadata: exact byte size plus LFS SHA256
for weight shards or Git blob SHA1 for ordinary files. A SHA256 is also recorded
for every file. Files retain `.partial` until successful verification. The final
`download_receipt.json` in the destination is written only after every file passes.
`source_metadata.json` preserves the source sizes and hashes.

To resume after a stopped process, first ensure no downloader is active, then run
the same script with the same output path; existing verified files are reused and
curl resumes partial files. Run under `nohup` with redirected stdin/stdout/stderr
for a detached download. The script uses the existing Python environment without
installing or changing dependencies:

```bash
/data/users/jiachangzhang/eb_vjepa/envs/eb_vjepa_pretrain/bin/python \
  /data/users/jiachangzhang/eb_vjepa/scripts/download_dream_vla.py \
  --output /data/users/jiachangzhang/eb_vjepa/checkpoints/dream_vla/Dream-VLA-7B
```

If mirror transport fails, `--endpoint https://huggingface.co --proxy
http://127.0.0.1:7890` selects the server's tested Mihomo HTTP proxy. Do not run
two downloaders against the same output directory.

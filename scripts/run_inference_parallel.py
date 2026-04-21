#!/usr/bin/env python3
"""Run inference.py in parallel across GPUs by splitting metadata entries."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

from omegaconf import OmegaConf


def split_round_robin(data, num_buckets):
    buckets = [[] for _ in range(num_buckets)]
    for idx, entry in enumerate(data):
        buckets[idx % num_buckets].append(entry)
    return buckets


def main():
    parser = argparse.ArgumentParser(description="Run PostCam inference in parallel across GPUs.")
    parser.add_argument("--metadata_json", required=True, help="Path to the metadata JSON file")
    parser.add_argument("--gpu_list", required=True, help="Comma-separated GPU IDs")
    parser.add_argument("--config", required=True, help="Base config path")
    parser.add_argument("--traj_txt_path", required=True, help="Trajectory txt path")
    parser.add_argument("--output_path", required=True, help="Output path")
    parser.add_argument("--resume_ckpt_path", required=True, help="Checkpoint path")
    parser.add_argument("--inference_script", required=True, help="Path to inference.py")
    args = parser.parse_args()

    with open(args.metadata_json, "r") as f:
        data = json.load(f)

    gpu_ids = [gpu.strip() for gpu in args.gpu_list.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("gpu_list must contain at least one GPU id")

    chunks = [chunk for chunk in split_round_robin(data, len(gpu_ids)) if chunk]
    active_gpu_ids = gpu_ids[:len(chunks)]
    if not chunks:
        print("No metadata entries found, nothing to run.")
        return

    tmp_dir = tempfile.mkdtemp(prefix="postcam_infer_parallel_")
    processes = []
    try:
        for worker_id, (gpu_id, chunk) in enumerate(zip(active_gpu_ids, chunks)):
            worker_json = os.path.join(tmp_dir, f"metadata_worker_{worker_id}.json")
            worker_config = os.path.join(tmp_dir, f"config_worker_{worker_id}.yaml")

            with open(worker_json, "w") as f:
                json.dump(chunk, f, indent=4, ensure_ascii=False)

            cfg = OmegaConf.load(args.config)
            cfg.dataset.metadata_paths = [worker_json]
            OmegaConf.save(cfg, worker_config)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            env["PYTHONUNBUFFERED"] = "1"
            cmd = [
                sys.executable,
                args.inference_script,
                "--config",
                worker_config,
                "--traj_txt_path",
                args.traj_txt_path,
                "--output_path",
                args.output_path,
                "--resume_ckpt_path",
                args.resume_ckpt_path,
            ]
            print(f"[worker {worker_id}] launching on GPU {gpu_id} with {len(chunk)} videos")
            processes.append((gpu_id, subprocess.Popen(cmd, env=env)))

        failed = []
        for gpu_id, proc in processes:
            ret = proc.wait()
            if ret != 0:
                failed.append((gpu_id, ret))

        if failed:
            for gpu_id, ret in failed:
                print(f"Inference worker on GPU {gpu_id} failed with code {ret}", file=sys.stderr)
            sys.exit(1)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

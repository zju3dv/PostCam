#!/bin/bash
set -e

COMMON_ARGS=(
  --input_dir ./test
  --step1_gpu 0,1
  --step2_gpu 0,1
  --step3_gpu 0,1
)

TRAJECTORIES=(
  y_left_30
  y_right_30
  x_up_30
  x_down_30
  zoom_in
  zoom_out
)

for idx in "${!TRAJECTORIES[@]}"; do
  traj_name="${TRAJECTORIES[$idx]}"
  extra_args=()
  if [ "$idx" -gt 0 ]; then
    extra_args+=(--skip_step1 --skip_step2)
  fi

  bash run_pipeline.sh \
    "${COMMON_ARGS[@]}" \
    --traj_txt_path "./traj/${traj_name}.txt" \
    "${extra_args[@]}"
done

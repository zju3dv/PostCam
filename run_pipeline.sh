#!/bin/bash
set -e

##############################################################################
# PostCam: One-stop inference pipeline
#   Step 1: Generate captions with Florence-2 + build JSON metadata
#   Step 2: Estimate depth & camera poses with DA3 + convert to pipeline format
#   Step 3: Run PostCam video generation
#
# Usage:
#   bash run_pipeline.sh \
#     --input_dir ./test_input \
#     --traj_txt_path ./traj/y_left_30.txt
#
# Arguments:
#   --input_dir           (required) Folder containing input .mp4 videos
#   --traj_txt_path       (required) Camera trajectory file, e.g. ./traj/y_left_30.txt
#   --checkpoint_path     (optional) PostCam checkpoint path
#                         Default: ./checkpoints/PostCam/postcam.ckpt
#   --config_path         (optional) Config file path, default ./inference.yaml
#   --da3_model_path      (optional) DA3 model path, default ./checkpoints/DA3
#   --florence_model_path (optional) Florence-2 model path, default ./checkpoints/Florence-2-large
#   --step1_gpu           (optional) GPU for Step 1, default 0
#   --step2_gpu          (optional) GPUs for Step 2, comma-separated (e.g. 0,1,2,3), default 0
#   --step3_gpu           (optional) GPU for Step 3, default 0
#   --output_dir          (optional) Output directory (default: ./output)
#   --skip_step1          (optional) Skip Step 1 (caption generation)
#   --skip_step2          (optional) Skip Step 2 (depth estimation)
#   --skip_step3          (optional) Skip Step 3 (PostCam inference)
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default arguments
STEP1_GPU="0"
step2_gpu="0"
STEP3_GPU="0"
CHECKPOINT_PATH="${SCRIPT_DIR}/checkpoints/PostCam/postcam.ckpt"
CONFIG_PATH="${SCRIPT_DIR}/inference.yaml"
FLORENCE_MODEL_PATH="${SCRIPT_DIR}/checkpoints/Florence-2-large"
DA3_MODEL_PATH="${SCRIPT_DIR}/checkpoints/DA3"
OUTPUT_DIR="${SCRIPT_DIR}/output"
SKIP_STEP1=false
SKIP_STEP2=false
SKIP_STEP3=false

split_gpu_list() {
    local gpu_list="$1"
    IFS=',' read -r -a GPU_IDS <<< "$gpu_list"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dir)        INPUT_DIR="$2";           shift 2 ;;
        --traj_txt_path)    TRAJ_TXT_PATH="$2";       shift 2 ;;
        --checkpoint_path)  CHECKPOINT_PATH="$2";      shift 2 ;;
        --config_path)      CONFIG_PATH="$2";          shift 2 ;;
        --florence_model_path) FLORENCE_MODEL_PATH="$2"; shift 2 ;;
        --da3_model_path)   DA3_MODEL_PATH="$2";       shift 2 ;;
        --step1_gpu)        STEP1_GPU="$2";            shift 2 ;;
        --step2_gpu)       step2_gpu="$2";           shift 2 ;;
        --step3_gpu)        STEP3_GPU="$2";            shift 2 ;;
        --output_dir)       OUTPUT_DIR="$2";           shift 2 ;;
        --skip_step1)       SKIP_STEP1=true;           shift ;;
        --skip_step2)       SKIP_STEP2=true;           shift ;;
        --skip_step3)       SKIP_STEP3=true;           shift ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$INPUT_DIR" ]; then
    echo "Error: --input_dir is required"
    exit 1
fi
if [ -z "$TRAJ_TXT_PATH" ]; then
    echo "Error: --traj_txt_path is required"
    exit 1
fi

INPUT_DIR_NAME=$(basename "$INPUT_DIR")
TRAJ_NAME=$(basename "$TRAJ_TXT_PATH" .txt)
JSON_PATH="${INPUT_DIR}/metadata.json"

echo "============================================"
echo "PostCam Pipeline Configuration:"
echo "  Input dir:       $INPUT_DIR"
echo "  Traj txt path:   $TRAJ_TXT_PATH"
echo "  JSON path:       $JSON_PATH"
echo "  Output dir:      $OUTPUT_DIR"
echo "  Step1 GPU:       $STEP1_GPU"
echo "  Step2 GPUs:      $step2_gpu"
echo "  Step3 GPU:       $STEP3_GPU"
echo "  Checkpoint:      $CHECKPOINT_PATH"
echo "  Config:          $CONFIG_PATH"
echo "  DA3 model:       $DA3_MODEL_PATH"
echo "  Florence model:  $FLORENCE_MODEL_PATH"
echo "============================================"

##############################################################################
# Step 1: Generate captions with Florence-2 + build JSON metadata
##############################################################################
if [ "$SKIP_STEP1" = false ]; then
    echo ""
    echo "========== Step 1: Generating captions with Florence-2 =========="

    split_gpu_list "$STEP1_GPU"
    if [ "${#GPU_IDS[@]}" -le 1 ]; then
        CUDA_VISIBLE_DEVICES=$STEP1_GPU python "$SCRIPT_DIR/scripts/gen_json.py" \
            --root_dir "$INPUT_DIR" \
            --model_path "$FLORENCE_MODEL_PATH"
    else
        PIDS=()
        for worker_id in "${!GPU_IDS[@]}"; do
            gpu_id="${GPU_IDS[$worker_id]}"
            partial_json="${INPUT_DIR}/new_partial_${worker_id}.json"
            CUDA_VISIBLE_DEVICES=$gpu_id python "$SCRIPT_DIR/scripts/gen_json.py" \
                --root_dir "$INPUT_DIR" \
                --model_path "$FLORENCE_MODEL_PATH" \
                --worker_id "$worker_id" \
                --num_workers "${#GPU_IDS[@]}" \
                --output_json "$partial_json" &
            PIDS+=($!)
        done

        STEP1_FAILED=0
        for pid in "${PIDS[@]}"; do
            if ! wait "$pid"; then
                STEP1_FAILED=1
            fi
        done
        if [ "$STEP1_FAILED" -ne 0 ]; then
            echo "Step 1 failed on at least one worker"
            exit 1
        fi

        python "$SCRIPT_DIR/scripts/merge_partial_jsons.py" \
            --input_dir "$INPUT_DIR" \
            --output_json "$JSON_PATH"
    fi

    echo "Step 1 completed. JSON saved to: $JSON_PATH"
else
    echo ""
    echo "========== Step 1: SKIPPED =========="
fi

##############################################################################
# Step 2: Depth estimation with DA3 + convert to pipeline format
##############################################################################
if [ "$SKIP_STEP2" = false ]; then
    echo ""
    echo "========== Step 2: DA3 depth estimation + format conversion =========="

    DA3_CLI="${SCRIPT_DIR}/depth/depth_predict_da3_cli.py"
    DA3_CONFIG="{\"model_path\":\"${DA3_MODEL_PATH}\",\"fix_resize\":true,\"fix_resize_height\":480,\"fix_resize_width\":832,\"num_frames\":1000,\"save_point_cloud\":true}"
    CONVERT_SCRIPT="${SCRIPT_DIR}/scripts/convert_da3_to_pi3.py"

    python "$SCRIPT_DIR/scripts/run_da3_parallel.py" \
        --json_path "$JSON_PATH" \
        --gpu_list "$step2_gpu" \
        --da3_cli "$DA3_CLI" \
        --da3_config "$DA3_CONFIG" \
        --convert_script "$CONVERT_SCRIPT"

    echo "Step 2 completed. Depth maps generated."
else
    echo ""
    echo "========== Step 2: SKIPPED =========="
fi

##############################################################################
# Step 3: PostCam video generation
##############################################################################
if [ "$SKIP_STEP3" = false ]; then
    echo ""
    echo "========== Step 3: Running PostCam inference =========="

    # Create a temporary config with updated paths
    TMP_CONFIG=$(mktemp /tmp/postcam_config_XXXXXX.yaml)
    cp "$CONFIG_PATH" "$TMP_CONFIG"

    # Override metadata_paths to point to our JSON
    sed -i "s|metadata_paths:.*|metadata_paths:|" "$TMP_CONFIG"
    sed -i "/metadata_paths:/a\\  - ${JSON_PATH}" "$TMP_CONFIG"
    # Remove the old list entry if it exists
    sed -i "/- \.\/examples\/test\.json/d" "$TMP_CONFIG"

    split_gpu_list "$STEP3_GPU"
    if [ "${#GPU_IDS[@]}" -le 1 ]; then
        CUDA_VISIBLE_DEVICES=$STEP3_GPU python "$SCRIPT_DIR/inference.py" \
            --config "$TMP_CONFIG" \
            --traj_txt_path "$TRAJ_TXT_PATH" \
            --output_path "$OUTPUT_DIR" \
            --resume_ckpt_path "$CHECKPOINT_PATH"
    else
        python "$SCRIPT_DIR/scripts/run_inference_parallel.py" \
            --metadata_json "$JSON_PATH" \
            --gpu_list "$STEP3_GPU" \
            --config "$TMP_CONFIG" \
            --traj_txt_path "$TRAJ_TXT_PATH" \
            --output_path "$OUTPUT_DIR" \
            --resume_ckpt_path "$CHECKPOINT_PATH" \
            --inference_script "$SCRIPT_DIR/inference.py"
    fi

    rm -f "$TMP_CONFIG"

    echo "Step 3 completed. Results saved to: $OUTPUT_DIR"
else
    echo ""
    echo "========== Step 3: SKIPPED =========="
fi

echo ""
echo "============================================"
echo "Pipeline finished!"
echo "  JSON:    $JSON_PATH"
echo "  Output:  $OUTPUT_DIR"
echo "============================================"

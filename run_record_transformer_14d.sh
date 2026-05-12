#!/bin/bash
set -euo pipefail

# ============================================================
# 14D Transformer rollout + screen recording script
# Saves:
#   1) one video per rollout
#   2) one combined video
#   3) one log per rollout
# ============================================================

PROJECT_DIR="/home/chetana/pir_project"
ENV_NAME="immimic"

NUM_ROLLOUTS=3
HORIZON=900
DEBUG_EVERY=20
GOAL_X=0.15
GOAL_Y=0.15
END_WAIT_SEC=5

# Set to 1 only if the cube drops too early after close.
HOLD_AFTER_FIRST_CLOSE=0

cd "$PROJECT_DIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

mkdir -p recordings logs

RUN_ID=$(date +%Y%m%d_%H%M%S)
OUT_DIR="recordings/transformer_14d_runs_${RUN_ID}"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "RUN ID: $RUN_ID"
echo "OUTPUT DIR: $OUT_DIR"
echo "DISPLAY: ${DISPLAY:-EMPTY}"
echo "============================================================"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: ffmpeg not found."
    echo "Install with: sudo apt install ffmpeg"
    exit 1
fi

if ! command -v xdpyinfo >/dev/null 2>&1; then
    echo "ERROR: xdpyinfo not found."
    echo "Install with: sudo apt install x11-utils"
    exit 1
fi

SCREEN_SIZE=$(xdpyinfo | awk '/dimensions/{print $2}')
echo "SCREEN_SIZE: $SCREEN_SIZE"

VIDEO_LIST="$OUT_DIR/video_list.txt"
rm -f "$VIDEO_LIST"

for i in $(seq 1 "$NUM_ROLLOUTS"); do
    IDX=$(printf "%02d" "$i")
    VIDEO_FILE="$OUT_DIR/rollout_${IDX}.mp4"
    LOG_FILE="logs/eval_transformer_14d_rollout_${IDX}_${RUN_ID}.log"

    echo "============================================================"
    echo "ROLLOUT $IDX / $NUM_ROLLOUTS"
    echo "VIDEO: $VIDEO_FILE"
    echo "LOG: $LOG_FILE"
    echo "============================================================"

    ffmpeg -y \
        -f x11grab \
        -framerate 30 \
        -video_size "$SCREEN_SIZE" \
        -i "$DISPLAY" \
        -vcodec libx264 \
        -preset ultrafast \
        -pix_fmt yuv420p \
        "$VIDEO_FILE" \
        > "$OUT_DIR/ffmpeg_rollout_${IDX}.log" 2>&1 &

    FFMPEG_PID=$!
    echo "Started ffmpeg PID: $FFMPEG_PID"

    sleep 2

    CMD=(
        python -u src/evaluation/eval_transformer_on_franka_14d.py
        --episodes 1
        --horizon "$HORIZON"
        --debug-every "$DEBUG_EVERY"
        --goal-x "$GOAL_X"
        --goal-y "$GOAL_Y"
        --end-wait-sec "$END_WAIT_SEC"
    )

    if [ "$HOLD_AFTER_FIRST_CLOSE" -eq 1 ]; then
        CMD+=(--hold-after-first-close)
    fi

    "${CMD[@]}" | tee "$LOG_FILE"

    echo "Stopping ffmpeg PID: $FFMPEG_PID"
    kill -INT "$FFMPEG_PID" || true
    wait "$FFMPEG_PID" || true

    echo "file '$(realpath "$VIDEO_FILE")'" >> "$VIDEO_LIST"

    sleep 2
done

COMBINED_VIDEO="$OUT_DIR/transformer_14d_all_rollouts_${RUN_ID}.mp4"

echo "============================================================"
echo "Combining videos..."
echo "============================================================"

ffmpeg -y \
    -f concat \
    -safe 0 \
    -i "$VIDEO_LIST" \
    -c copy \
    "$COMBINED_VIDEO" \
    > "$OUT_DIR/ffmpeg_concat.log" 2>&1 || {
        echo "Concat copy failed. Retrying with re-encode..."
        ffmpeg -y \
            -f concat \
            -safe 0 \
            -i "$VIDEO_LIST" \
            -vcodec libx264 \
            -pix_fmt yuv420p \
            "$COMBINED_VIDEO"
    }

echo "============================================================"
echo "DONE"
echo "Saved rollout videos in:"
echo "$OUT_DIR"
echo ""
echo "Combined video:"
echo "$COMBINED_VIDEO"
echo ""
echo "Logs:"
ls -lh logs/eval_transformer_14d_rollout_*_"$RUN_ID".log
echo ""
echo "Videos:"
ls -lh "$OUT_DIR"
echo "============================================================"

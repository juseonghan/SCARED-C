#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <root> <output_root>"
    exit 1
fi

ROOT="$(realpath "$1")"
OUTPUT_ROOT="$(realpath -m "$2")"

DATASETS=("dataset_7")
KEYFRAMES=("keyframe_4")

# How often to snapshot the mapper state (every N registered images).
# Lower = more frequent saves = safer against timeouts, but slightly more I/O.
# With ~1000 frames you might set this to 50-100; with fewer frames, 10-25.
SNAPSHOT_IMAGES_FREQ=50

# =============================================================================
# SLURM-aware graceful shutdown
# =============================================================================
# SLURM sends SIGTERM (then SIGKILL after --signal grace period) before the
# wall-time limit.  We trap it so the mapper can finish its current step and
# the snapshot on disk stays consistent.
CAUGHT_SIGNAL=0
trap 'echo "[SIGNAL] Caught SIGTERM/SIGINT — will exit after current step."; CAUGHT_SIGNAL=1' SIGTERM SIGINT

# =============================================================================
# Pipeline function (runs one sequence)
# =============================================================================
run_pipeline() {
    local SEQ_DIR="$1"
    local OUTPUT_DIR="$2"
    local IMAGES_OUT="$OUTPUT_DIR/images"
    local CHECKPOINT_DIR="$OUTPUT_DIR/.checkpoints"

    # --- Validate inputs ---
    if [[ ! -f "$SEQ_DIR/Left_Image.png" ]]; then
        echo "  ERROR: $SEQ_DIR/Left_Image.png not found — skipping."
        return 1
    fi
    if [[ ! -f "$SEQ_DIR/data/rgb.mp4" ]]; then
        echo "  ERROR: $SEQ_DIR/data/rgb.mp4 not found — skipping."
        return 1
    fi

    # --- Prepare directories ---
    mkdir -p "$IMAGES_OUT"
    mkdir -p "$OUTPUT_DIR/sparse"
    mkdir -p "$CHECKPOINT_DIR"

    # --- Checkpoint helpers ---
    _ckpt_exists() { [[ -f "$CHECKPOINT_DIR/$1.done" ]]; }
    _ckpt_mark()  { date -Iseconds > "$CHECKPOINT_DIR/$1.done"; echo "    -> Checkpoint '$1' saved."; }

    # ----- Stage 1: Copy keyframe -----
    if _ckpt_exists "keyframe"; then
        echo "  [SKIP] 1/5 Keyframe"
    else
        echo "  [RUN]  1/5 Copying Left_Image.png -> keyframe.png"
        cp "$SEQ_DIR/Left_Image.png" "$IMAGES_OUT/keyframe.png"
        _ckpt_mark "keyframe"
    fi

    [[ $CAUGHT_SIGNAL -eq 1 ]] && { echo "  Exiting early (signal)."; return 1; }

    # ----- Stage 2: Extract top-half frames -----
    if _ckpt_exists "frames_extracted"; then
        echo "  [SKIP] 2/5 Frame extraction"
    else
        echo "  [RUN]  2/5 Extracting top-half frames from rgb.mp4 ..."
        local FRAMES_TMP="$OUTPUT_DIR/.frames_tmp"
        rm -rf "$FRAMES_TMP"
        mkdir -p "$FRAMES_TMP"

        ffmpeg -i "$SEQ_DIR/data/rgb.mp4" \
            -vf "crop=iw:ih/2:0:0" \
            "$FRAMES_TMP/frame_%06d.png" \
            -y -loglevel warning

        mv "$FRAMES_TMP"/frame_*.png "$IMAGES_OUT/"
        rm -rf "$FRAMES_TMP"

        echo "    Extracted $(ls "$IMAGES_OUT"/frame_*.png 2>/dev/null | wc -l) frames"
        _ckpt_mark "frames_extracted"
    fi

    [[ $CAUGHT_SIGNAL -eq 1 ]] && { echo "  Exiting early (signal)."; return 1; }

    # ----- Stage 3: Feature extraction -----
    if _ckpt_exists "feature_extraction"; then
        echo "  [SKIP] 3/5 Feature extraction"
    else
        echo "  [RUN]  3/5 COLMAP feature_extractor ..."
        colmap feature_extractor \
            --database_path "$OUTPUT_DIR/database.db" \
            --image_path "$IMAGES_OUT" \
            --ImageReader.camera_model PINHOLE \
            --ImageReader.single_camera 1 \
            --ImageReader.camera_params "1073.64844,1073.43433,586.080322,512.475891" \
            --FeatureExtraction.use_gpu 0

        _ckpt_mark "feature_extraction"
    fi

    [[ $CAUGHT_SIGNAL -eq 1 ]] && { echo "  Exiting early (signal)."; return 1; }

    # ----- Stage 4: Exhaustive matching -----
    if _ckpt_exists "exhaustive_matching"; then
        echo "  [SKIP] 4/5 Exhaustive matching"
    else
        echo "  [RUN]  4/5 COLMAP exhaustive_matcher ..."
        colmap exhaustive_matcher \
            --database_path "$OUTPUT_DIR/database.db" \
            --FeatureMatching.use_gpu 0

        _ckpt_mark "exhaustive_matching"
    fi

    [[ $CAUGHT_SIGNAL -eq 1 ]] && { echo "  Exiting early (signal)."; return 1; }

    # ----- Stage 5: Mapper (with snapshot checkpointing + resume) -----
    if _ckpt_exists "mapper"; then
        echo "  [SKIP] 5/5 Mapper"
    else
        local SNAPSHOT_DIR="$OUTPUT_DIR/snapshots"
        mkdir -p "$SNAPSHOT_DIR"

        # ---- Find the latest snapshot to resume from (if any) ----
        # COLMAP snapshots are saved as numbered subdirectories inside
        # snapshot_path, e.g. snapshots/0/, snapshots/1/, etc.
        # Each contains cameras.bin, images.bin, points3D.bin.
        # We find the highest-numbered valid snapshot.
        local RESUME_PATH=""
        if [[ -d "$SNAPSHOT_DIR" ]]; then
            local LATEST_SNAP=""
            for SNAP_SUBDIR in "$SNAPSHOT_DIR"/*/; do
                # Check it actually contains a reconstruction
                if [[ -f "${SNAP_SUBDIR}cameras.bin" ]] || [[ -f "${SNAP_SUBDIR}cameras.txt" ]]; then
                    LATEST_SNAP="$SNAP_SUBDIR"
                fi
            done
            if [[ -n "$LATEST_SNAP" ]]; then
                RESUME_PATH="$LATEST_SNAP"
            fi
        fi

        if [[ -n "$RESUME_PATH" ]]; then
            echo "  [RUN]  5/5 COLMAP mapper (RESUMING from snapshot: $RESUME_PATH) ..."
            colmap mapper \
                --database_path "$OUTPUT_DIR/database.db" \
                --image_path "$IMAGES_OUT" \
                --output_path "$OUTPUT_DIR/sparse" \
                --input_path "$RESUME_PATH" \
                --Mapper.ba_use_gpu 0 \
                --Mapper.snapshot_path "$SNAPSHOT_DIR" \
                --Mapper.snapshot_frames_freq "$SNAPSHOT_IMAGES_FREQ"
        else
            echo "  [RUN]  5/5 COLMAP mapper (fresh start) ..."
            colmap mapper \
                --database_path "$OUTPUT_DIR/database.db" \
                --image_path "$IMAGES_OUT" \
                --output_path "$OUTPUT_DIR/sparse" \
                --Mapper.ba_use_gpu 0 \
                --Mapper.snapshot_path "$SNAPSHOT_DIR" \
                --Mapper.snapshot_frames_freq "$SNAPSHOT_IMAGES_FREQ"
        fi

        _ckpt_mark "mapper"
    fi

    echo "  Done."
}

# =============================================================================
# Main loop
# =============================================================================
TOTAL=$(( ${#DATASETS[@]} * ${#KEYFRAMES[@]} ))
COUNT=0

echo "========================================="
echo " COLMAP batch pipeline"
echo " Root:       $ROOT"
echo " Output:     $OUTPUT_ROOT"
echo " Sequences:  $TOTAL"
echo " Snapshot freq: every $SNAPSHOT_IMAGES_FREQ registered images"
echo "========================================="
echo ""

for DATASET in "${DATASETS[@]}"; do
    for KEYFRAME in "${KEYFRAMES[@]}"; do
        COUNT=$((COUNT + 1))
        SEQ_DIR="$ROOT/$DATASET/$KEYFRAME"
        OUT_DIR="$OUTPUT_ROOT/$DATASET/$KEYFRAME"

        echo "[$COUNT/$TOTAL] $DATASET / $KEYFRAME"
        echo "  Seq: $SEQ_DIR"
        echo "  Out: $OUT_DIR"

        run_pipeline "$SEQ_DIR" "$OUT_DIR" || true

        # If we caught a signal, stop processing further sequences
        if [[ $CAUGHT_SIGNAL -eq 1 ]]; then
            echo ""
            echo "[SIGNAL] Stopping batch loop due to signal."
            break 2
        fi

        echo ""
    done
done

echo "========================================="
if [[ $CAUGHT_SIGNAL -eq 1 ]]; then
    echo " Batch interrupted by signal. $COUNT/$TOTAL sequences attempted."
    echo " Re-submit the job to resume from the latest snapshots."
else
    echo " Batch complete. $COUNT/$TOTAL sequences processed."
fi
echo "========================================="
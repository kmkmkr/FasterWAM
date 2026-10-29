#!/usr/bin/env bash
# Run inside an existing FasterWAM environment on a GPU server. No job submission.
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Required env: TRAIN_DATASET VAL_DATASET NORM_STATS TEXT_CACHE MODEL_ROOT OUTPUT_ROOT
Optional: NPROC=1 MAX_STEPS=5 BATCH_SIZE=1 NUM_WORKERS=0 SEED=42 SIMULATED_DELAY=8
          INIT_CHECKPOINT=/path/to/weights.pt ACTION_DIT_PATH=/path/to/backbone.pt
          RTC_PYTHON=python RUN_ID=<unique-name>
Usage: bash scripts/train_widowxai_training_rtc.sh [--dry-run] [Hydra overrides...]
--dry-run validates inputs and saves the resolved config/manifest without training.
Every invocation requires a new run directory; reuse with RUN_ID is rejected.
HELP
  exit 0
fi

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=true; shift; fi
: "${TRAIN_DATASET:?Set TRAIN_DATASET to an existing LeRobot dataset}"
: "${VAL_DATASET:?Set VAL_DATASET to an existing held-out LeRobot dataset}"
: "${NORM_STATS:?Set NORM_STATS to the matching dataset_stats.json}"
: "${TEXT_CACHE:?Set TEXT_CACHE to the existing task embedding directory}"
: "${MODEL_ROOT:?Set MODEL_ROOT to the local Wan model root}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a writable experiment directory}"
RTC_PYTHON="${RTC_PYTHON:-python}"
NPROC="${NPROC:-1}"
RUN_ID="${RUN_ID:-training_rtc_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

for required_file in "${TRAIN_DATASET}/meta/info.json" "${VAL_DATASET}/meta/info.json" "${NORM_STATS}"; do
  [[ -s "${required_file}" ]] || { echo "Missing required file: ${required_file}" >&2; exit 1; }
done
for required_directory in "${TEXT_CACHE}" "${MODEL_ROOT}"; do
  [[ -d "${required_directory}" ]] || { echo "Missing directory: ${required_directory}" >&2; exit 1; }
done
if [[ -n "${INIT_CHECKPOINT:-}" && ! -s "${INIT_CHECKPOINT}" ]]; then
  echo "INIT_CHECKPOINT is missing: ${INIT_CHECKPOINT}" >&2; exit 1
fi
if [[ -n "${ACTION_DIT_PATH:-}" && ! -s "${ACTION_DIT_PATH}" ]]; then
  echo "ACTION_DIT_PATH is missing: ${ACTION_DIT_PATH}" >&2; exit 1
fi
if [[ ! "${NPROC}" =~ ^[1-9][0-9]*$ || ! "${RUN_ID}" =~ ^[A-Za-z0-9_.-]+$ || "${RUN_ID}" == "." || "${RUN_ID}" == ".." ]]; then
  echo "Invalid NPROC or RUN_ID" >&2; exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_ROOT}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

mkdir -p -- "${OUTPUT_ROOT}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
mkdir -- "${RUN_DIR}"  # Exclusive creation protects all prior runs.
RUN_DIR="$(cd -- "${RUN_DIR}" && pwd)"
TRAIN_ARGS=(
  "task=widowxai_fasterwam_2wrist_training_rtc"
  "output_dir=${RUN_DIR}"
  "data.train.dataset_dirs=[${TRAIN_DATASET}]"
  "data.val.dataset_dirs=[${VAL_DATASET}]"
  "data.train.pretrained_norm_stats=${NORM_STATS}"
  "data.val.pretrained_norm_stats=${NORM_STATS}"
  "data.train.text_embedding_cache_dir=${TEXT_CACHE}"
  "data.val.text_embedding_cache_dir=${TEXT_CACHE}"
  "model.redirect_common_files=false"
  "model.training_rtc.simulated_delay=${SIMULATED_DELAY:-8}"
  "batch_size=${BATCH_SIZE:-1}"
  "num_workers=${NUM_WORKERS:-0}"
  "max_steps=${MAX_STEPS:-5}"
  "num_epochs=1"
  "gradient_accumulation_steps=1"
  "log_every=1" "save_every=${MAX_STEPS:-5}" "eval_every=0"
  "keep_last_n_states=null" "seed=${SEED:-42}" "wandb.enabled=false"
  "resume=${INIT_CHECKPOINT:-null}"
  "model.action_dit_pretrained_path=${ACTION_DIT_PATH:-null}"
)
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  # A full FasterWAM weight checkpoint replaces every random DiT parameter.
  TRAIN_ARGS+=("model.skip_dit_load_from_pretrain=true")
fi
TRAIN_ARGS+=("$@")

"${RTC_PYTHON}" scripts/train.py "${TRAIN_ARGS[@]}" --cfg job --resolve > "${RUN_DIR}/resolved_config.yaml"
"${RTC_PYTHON}" - "${RUN_DIR}" "${NPROC}" "${TRAIN_ARGS[@]}" <<'PY'
import datetime
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from omegaconf import OmegaConf

run = Path(sys.argv[1])
cfg = OmegaConf.load(run / "resolved_config.yaml")
if Path(str(cfg.output_dir)).resolve() != run:
    raise ValueError("Override output_dir via OUTPUT_ROOT/RUN_ID, not a Hydra override")
if cfg.keep_last_n_states is not None:
    raise ValueError("This launcher preserves all experiment states; keep_last_n_states must be null")
if cfg.wandb.enabled:
    raise ValueError("This launcher uses offline logging; wandb.enabled must be false")
delay = cfg.model.training_rtc.simulated_delay
if type(delay) is not int or not 1 <= delay < int(cfg.data.train.num_frames):
    raise ValueError("simulated_delay must be in [1, action_horizon]")

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

datasets = []
for split in ("train", "val"):
    for directory in cfg.data[split].dataset_dirs:
        root = Path(directory)
        meta = {str(p.relative_to(root)): sha256(p) for p in sorted((root / "meta").glob("*.json*"))}
        datasets.append({"split": split, "path": str(root.resolve()), "metadata_sha256": meta})
payload = {
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "code_status": subprocess.check_output(["git", "status", "--short"], text=True),
    "python": platform.python_version(), "seed": cfg.seed, "num_processes": int(sys.argv[2]),
    "model": cfg.model._target_, "base_model": cfg.model.model_id,
    "initial_checkpoint": cfg.resume,
    "initial_checkpoint_sha256": sha256(cfg.resume) if cfg.resume and Path(str(cfg.resume)).is_file() else None,
    "normalization_sha256": sha256(cfg.data.train.pretrained_norm_stats),
    "datasets": datasets,
    "text_cache": str(cfg.data.train.text_embedding_cache_dir),
    "note": "Dataset fingerprints cover metadata, not the raw videos/parquet files.",
    "hydra_arguments": sys.argv[3:],
}
(run / "launch_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
print(f"[training-rtc] run={run} simulated_delay={delay} max_steps={cfg.max_steps}")
PY
printf '%q ' "${RTC_PYTHON}" -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_offload_ds.yaml \
  --num_processes "${NPROC}" scripts/train.py "${TRAIN_ARGS[@]}" > "${RUN_DIR}/command.txt"
printf '\n' >> "${RUN_DIR}/command.txt"
if "${DRY_RUN}"; then
  echo "[training-rtc] dry-run complete; no model load or training: ${RUN_DIR}"
  exit 0
fi
"${RTC_PYTHON}" -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_offload_ds.yaml \
  --num_processes "${NPROC}" scripts/train.py "${TRAIN_ARGS[@]}" \
  2>&1 | tee "${RUN_DIR}/train.log"

#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

CONDA_BASE="${CONDA_BASE:-/home/user/anaconda3}"
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi

conda activate SFP_ovss

timestamp="$(date +%Y%m%d_%H%M%S)"
work_root="${WORK_ROOT:-${REPO_DIR}/work_logs_tf_ovdg_original/${timestamp}}"
mkdir -p "${work_root}"

configs=(
  "acdc19_orig:configs/cfg_acdc19_orig.py"
  "acdc41_orig:configs/cfg_acdc41_orig.py"
  "bdd19_orig:configs/cfg_bdd19_orig.py"
  "bdd41_orig:configs/cfg_bdd41_orig.py"
  "mapi19_orig:configs/cfg_mapi19_orig.py"
  "mapi30_orig:configs/cfg_mapi30_orig.py"
  "roadwork10_orig:configs/cfg_roadwork10_orig.py"
)

gpus=(0 1 2 3)

find_free_gpu () {
  for gpu in "${gpus[@]}"; do
    if [[ -z "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]; then
      echo "${gpu}"
      return 0
    fi
  done
  return 1
}

launch_job () {
  local name="$1"
  local cfg="$2"
  local gpu="$3"
  local out_dir="${work_root}/${name}"
  mkdir -p "${out_dir}"

  echo "[${name}] start on GPU ${gpu} -> ${out_dir}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
    python eval.py --config "${cfg}" --work-dir "${out_dir}"
  ) >"${out_dir}/stdout.log" 2>"${out_dir}/stderr.log" &
}

running=0
idx=0

while [[ "${running}" -lt "${#gpus[@]}" && "${idx}" -lt "${#configs[@]}" ]]; do
  IFS=':' read -r name cfg <<<"${configs[$idx]}"
  launch_job "${name}" "${cfg}" "${gpus[$running]}"
  running=$((running+1))
  idx=$((idx+1))
done

while [[ "${idx}" -lt "${#configs[@]}" ]]; do
  wait -n
  free_gpu=""
  until free_gpu="$(find_free_gpu)"; do
    sleep 2
  done
  IFS=':' read -r name cfg <<<"${configs[$idx]}"
  launch_job "${name}" "${cfg}" "${free_gpu}"
  idx=$((idx+1))
done

wait
echo "All ORIGINAL-vocab benchmarks completed. Results under: ${work_root}"


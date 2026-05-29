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
work_root="${WORK_ROOT:-${REPO_DIR}/work_logs_tf_ovdg/${timestamp}}"
mkdir -p "${work_root}"

configs=(
  "acdc19:configs/cfg_acdc19.py"
  "acdc41:configs/cfg_acdc41.py"
  "bdd19:configs/cfg_bdd19.py"
  "bdd41:configs/cfg_bdd41.py"
  "mapi19:configs/cfg_mapi19.py"
  "mapi30:configs/cfg_mapi30.py"
  "roadwork10:configs/cfg_roadwork10.py"
)

gpus=(0 1 2 3)

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

find_free_gpu () {
  for gpu in "${gpus[@]}"; do
    # empty output means no compute process on this GPU
    if [[ -z "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]; then
      echo "${gpu}"
      return 0
    fi
  done
  return 1
}

running=0
idx=0

# Launch initial wave
while [[ "${running}" -lt "${#gpus[@]}" && "${idx}" -lt "${#configs[@]}" ]]; do
  IFS=':' read -r name cfg <<<"${configs[$idx]}"
  launch_job "${name}" "${cfg}" "${gpus[$running]}"
  running=$((running+1))
  idx=$((idx+1))
done

# Schedule remaining benchmarks as GPUs free up
while [[ "${idx}" -lt "${#configs[@]}" ]]; do
  # Wait for any one job to finish, then pick an actually-free GPU.
  wait -n
  free_gpu=""
  until free_gpu="$(find_free_gpu)"; do
    sleep 2
  done

  IFS=':' read -r name cfg <<<"${configs[$idx]}"
  launch_job "${name}" "${cfg}" "${free_gpu}"
  idx=$((idx+1))
done

# Wait for all remaining
wait

echo "All benchmarks completed. Results under: ${work_root}"


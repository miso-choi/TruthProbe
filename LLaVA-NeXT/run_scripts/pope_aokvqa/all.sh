SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TRUTH_SCORES_DIR="${REPO_ROOT}/truth_scores"
export LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-${REPO_ROOT}/data}"

bash run_scripts/pope_aokvqa/llava15.sh
bash run_scripts/pope_aokvqa/llava_next.sh
bash run_scripts/pope_aokvqa/qwen2_5_vl_instruct.sh
bash run_scripts/pope_aokvqa/qwen2_5_vl_omni.sh
#!/usr/bin/env bash
# 依次跑：未改进基线 -> 5 个创新点分支，把每轮的最终 mAP 以及相对基线的提升
# (差值 / 百分比) 写入同一个结果文件 result/ablation_results.csv。
#
# 用法:
#   nohup ./run_ablation.sh > result/run_ablation.log 2>&1 &
#
# 注意：本脚本及 result/ 目录均未加入 git 版本管理（保持 untracked），
# 这样脚本内部 git checkout 切分支时不会把脚本自己删掉/改掉。

set -uo pipefail

# ========== 可配置项 ==========
GPU_ID=0
CONDA_ENV_NAME="SHC"
CODE_LENGTH=16
DATA_ROOT="./data/cifar-100-python/"
WAIT_SECONDS=600   # 每轮跑完后等待 10 分钟，让机器恢复原状
ORIGINAL_REF="feature-baseline-plan"
RESULT_DIR="result"
RESULT_FILE="$RESULT_DIR/ablation_results.csv"   # 最终结果文件（含对比列）
# ================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$RESULT_DIR/logs"
mkdir -p "$LOG_DIR"

# ========== 激活 conda 环境 ==========
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
else
    echo "[WARN] 未找到 conda 命令，假设当前 shell 已经是训练所需的 python 环境" >&2
fi

BASE_CMD="python run.py --code-length $CODE_LENGTH --root $DATA_ROOT --gpu $GPU_ID"

# stage 定义: name|git_ref|extra_args
STAGES=(
    "baseline|1fb8b75|"
    "attn|feature-baseline-plan-attn|"
    "bitreg|feature-baseline-plan-bitreg|--alpha-bal 0.01 --alpha-ind 0.001"
    "contrast|feature-baseline-plan-contrast|--alpha-cont 0.1 --tau 0.07"
    "refine|feature-baseline-plan-refine|--refine-interval 20 --refine-momentum 0.995"
    "diffusion|feature-baseline-plan-diffusion|--use-diffusion True --diff-alpha 0.15 --diff-steps 3 --sim-topk 0"
)

if [[ ! -f "$RESULT_FILE" ]]; then
    echo "stage,git_ref,extra_args,command,mAP_ALL,mAP_100,mAP_1000,delta_ALL,delta_ALL_pct,delta_100,delta_100_pct,delta_1000,delta_1000_pct,start_time,end_time,duration_min,log_file,status" > "$RESULT_FILE"
fi

csv_escape() {
    printf '%s' "$1" | sed 's/"/""/g'
}

# 计算 (当前值 - 基线值) 以及百分比，无法计算时输出 NA
calc_delta() {
    local cur="$1" base="$2"
    if [[ -z "$cur" || -z "$base" ]]; then
        echo "NA"
        return
    fi
    awk -v a="$cur" -v b="$base" 'BEGIN{printf "%+.5f", a-b}'
}

calc_pct() {
    local cur="$1" base="$2"
    if [[ -z "$cur" || -z "$base" ]]; then
        echo "NA"
        return
    fi
    awk -v a="$cur" -v b="$base" 'BEGIN{ if (b==0) {print "NA"} else {printf "%+.2f%%", (a-b)/b*100} }'
}

BASE_MAP_ALL=""
BASE_MAP_100=""
BASE_MAP_1000=""

for stage_def in "${STAGES[@]}"; do
    IFS='|' read -r STAGE_NAME GIT_REF EXTRA_ARGS <<< "$stage_def"

    echo "=============================================="
    echo "[$(date '+%F %T')] 开始 stage: $STAGE_NAME (ref=$GIT_REF)"
    echo "=============================================="

    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "[ERROR] 工作区有未提交的改动，为避免 checkout 冲突，脚本中止。" >&2
        exit 1
    fi

    if ! git checkout "$GIT_REF" 2>>"$LOG_DIR/git_checkout.log"; then
        echo "[ERROR] git checkout $GIT_REF 失败，跳过该 stage" >&2
        echo "$STAGE_NAME,$GIT_REF,\"$(csv_escape "$EXTRA_ARGS")\",,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,,checkout_failed" >> "$RESULT_FILE"
        continue
    fi

    FULL_CMD="$BASE_CMD $EXTRA_ARGS"
    LOG_FILE="$LOG_DIR/${STAGE_NAME}.log"
    START_TIME="$(date '+%F %T')"
    START_TS=$(date +%s)

    echo "[$(date '+%F %T')] 运行命令: $FULL_CMD"
    # shellcheck disable=SC2086
    eval $FULL_CMD 2>&1 | tee "$LOG_FILE"
    RUN_STATUS=${PIPESTATUS[0]}

    END_TS=$(date +%s)
    END_TIME="$(date '+%F %T')"
    DURATION_MIN=$(( (END_TS - START_TS) / 60 ))

    if [[ "$RUN_STATUS" -ne 0 ]]; then
        echo "[WARN] stage $STAGE_NAME 训练进程返回非零状态码 $RUN_STATUS，仍尝试解析日志" >&2
    fi

    MAP_ALL=$(grep -oE 'final_mAP_ALL:[0-9.]+' "$LOG_FILE" | tail -1 | cut -d: -f2)
    MAP_100=$(grep -oE 'final_mAP_100:[0-9.]+' "$LOG_FILE" | tail -1 | cut -d: -f2)
    MAP_1000=$(grep -oE 'final_mAP_1000:[0-9.]+' "$LOG_FILE" | tail -1 | cut -d: -f2)

    if [[ -z "$MAP_ALL" || -z "$MAP_100" || -z "$MAP_1000" ]]; then
        echo "[WARN] 未能从日志中解析出 mAP 指标: $LOG_FILE" >&2
        ROW_STATUS="parse_failed"
        MAP_ALL=${MAP_ALL:-NA}; MAP_100=${MAP_100:-NA}; MAP_1000=${MAP_1000:-NA}
    else
        ROW_STATUS="ok"
    fi

    if [[ "$STAGE_NAME" == "baseline" ]]; then
        BASE_MAP_ALL="$MAP_ALL"
        BASE_MAP_100="$MAP_100"
        BASE_MAP_1000="$MAP_1000"
        DELTA_ALL="+0.00000"; DELTA_ALL_PCT="+0.00%"
        DELTA_100="+0.00000"; DELTA_100_PCT="+0.00%"
        DELTA_1000="+0.00000"; DELTA_1000_PCT="+0.00%"
    else
        DELTA_ALL=$(calc_delta "$MAP_ALL" "$BASE_MAP_ALL")
        DELTA_ALL_PCT=$(calc_pct "$MAP_ALL" "$BASE_MAP_ALL")
        DELTA_100=$(calc_delta "$MAP_100" "$BASE_MAP_100")
        DELTA_100_PCT=$(calc_pct "$MAP_100" "$BASE_MAP_100")
        DELTA_1000=$(calc_delta "$MAP_1000" "$BASE_MAP_1000")
        DELTA_1000_PCT=$(calc_pct "$MAP_1000" "$BASE_MAP_1000")
    fi

    echo "$STAGE_NAME,$GIT_REF,\"$(csv_escape "$EXTRA_ARGS")\",\"$(csv_escape "$FULL_CMD")\",$MAP_ALL,$MAP_100,$MAP_1000,$DELTA_ALL,$DELTA_ALL_PCT,$DELTA_100,$DELTA_100_PCT,$DELTA_1000,$DELTA_1000_PCT,$START_TIME,$END_TIME,$DURATION_MIN,$LOG_FILE,$ROW_STATUS" >> "$RESULT_FILE"

    echo "[$(date '+%F %T')] stage $STAGE_NAME 完成: mAP_ALL=$MAP_ALL(Δ$DELTA_ALL_PCT) mAP_100=$MAP_100(Δ$DELTA_100_PCT) mAP_1000=$MAP_1000(Δ$DELTA_1000_PCT) (用时 ${DURATION_MIN} 分钟)"

    echo "[$(date '+%F %T')] 等待 ${WAIT_SECONDS}s 让机器恢复原状..."
    sleep "$WAIT_SECONDS"
done

git checkout "$ORIGINAL_REF" 2>>"$LOG_DIR/git_checkout.log"
echo "=============================================="
echo "[$(date '+%F %T')] 全部 stage 跑完，已切回 $ORIGINAL_REF"
echo "完整结果（含相对基线的提升指标）见: $RESULT_FILE"
echo "=============================================="

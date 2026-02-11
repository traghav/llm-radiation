#!/bin/bash
# Overnight batch runner - chains experiments sequentially
# Estimated total: ~12-14 hours on GH200
#
# Already running (will finish on their own):
#   - smoke_test_v2 (TinyLlama FP16) ~2h
#   - gptq_tinyllama_sweep ~2h
#   - gemma_2b_sweep ~2.5h
#
# This script runs AFTER those finish:
#   1. Gemma-7B FP16 (~5h)
#   2. Mistral-7B FP16 (~5h)
#   3. Phi-3-mini FP16 (~2.5h)

set -e

# Load tokens from .env file or environment
if [ -f /home/ubuntu/llm-radiation/.env.secrets ]; then
    source /home/ubuntu/llm-radiation/.env.secrets
fi
# Expects WANDB_API_KEY and HF_TOKEN to be set
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY in .env.secrets}"
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN in .env.secrets}"

cd /home/ubuntu/llm-radiation
LOGDIR=logs
mkdir -p "$LOGDIR"

UV="/home/ubuntu/.local/bin/uv"
RUN="$UV run --no-sync python3 scripts/run_experiment.py"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGDIR/overnight_batch.log"
}

run_experiment() {
    local config="$1"
    local name="$2"
    local logfile="$LOGDIR/${name}.log"

    log "=========================================="
    log "STARTING: $name"
    log "Config: $config"
    log "Log: $logfile"
    log "=========================================="

    if $RUN "$config" --log-level INFO > "$logfile" 2>&1; then
        log "COMPLETED: $name"
    else
        log "FAILED: $name (exit code $?)"
        log "Check $logfile for details"
    fi
    log ""
}

log "Overnight batch started"
log "Waiting for current experiments to finish..."

# Wait for GPTQ TinyLlama and Gemma-2B to finish
while pgrep -f "gptq_tinyllama_sweep|gemma_2b_sweep" > /dev/null 2>&1; do
    sleep 60
done
log "Previous experiments finished, starting overnight batch"

# 1. Gemma-7B (~5h)
run_experiment configs/sweeps/gemma_7b_sweep.yaml gemma_7b_sweep

# 2. Mistral-7B (~5h)
run_experiment configs/sweeps/mistral_7b_sweep.yaml mistral_7b_sweep

# 3. Phi-3-mini (~2.5h)
run_experiment configs/sweeps/phi3_mini_sweep.yaml phi3_mini_sweep

log "=========================================="
log "ALL EXPERIMENTS COMPLETE"
log "=========================================="
log "Results saved in results/ directory"
log "W&B dashboard: https://wandb.ai/llm-radiation"

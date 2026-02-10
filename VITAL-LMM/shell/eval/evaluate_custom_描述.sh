#!/usr/bin/env bash

# Deprecated wrapper for custom description.
# Inputs are configured in ./shell/eval/eval_data/internvl_eval_custom_conversation.json used by the canonical script.
# Expected environment variables: GPUS, BATCH_SIZE, PER_DEVICE_BATCH_SIZE (plus optional runtime CUDA/model variables).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/evaluate_custom_description.sh" "$@"

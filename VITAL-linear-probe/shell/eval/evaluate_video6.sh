set -x

GPUS=${GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GPUS_DEVICE=${GPUS_DEVICE:-0}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/jiaziheng/LMMS/internvl-pretrain-10_9_clip}"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=$(( $(date +%s) % 65535 ))  # Random port based on current system time
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='/dev/shm/internvl3_8b_dynamic_res_2nd_finetune_full_add_motion_val'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 4
# batch size per gpu: 1
# gradient accumulation steps: 4
# total batch size: 16
# epoch: 1
CUDA_VISIBLE_DEVICES=${GPUS_DEVICE} torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/eval/internvl_chat_finetune.py \
  --model_name_or_path ${MODEL_PATH} \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "/mnt/shared-storage-user/jiaziheng/pretrain/internvl-sft/internvl_chat/shell/eval/eval_data/internvl_eval_lsvqtest.json" \
 --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 12 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 16384 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
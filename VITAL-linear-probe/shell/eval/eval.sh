#!/bin/bash
# 主脚本：同时运行多个子脚本

# 获取当前脚本所在目录（防止路径错误）
BASE_DIR=$(dirname "$0")
MODEL_PATH="/mnt/shared-storage-user/jiaziheng/LMMS/internvl-mix-fidelity-11_29"
GPUS_DEVICE=0
# 子脚本路径
SCRIPT1="/mnt/shared-storage-user/jiaziheng/pretrain/internvl-sft/internvl_chat/shell/eval/evaluate_image1.sh"
SCRIPT2="/mnt/shared-storage-user/jiaziheng/pretrain/internvl-sft/internvl_chat/shell/eval/evaluate_video1.sh"
SCRIPT3="/mnt/shared-storage-user/jiaziheng/pretrain/internvl-sft/internvl_chat/shell/eval/evaluate_video2.sh"
SCRIPT4="/mnt/shared-storage-user/jiaziheng/pretrain/internvl-sft/internvl_chat/shell/eval/evaluate_video3.sh"
SCRIPT5="/mnt/shared-storage-user/jiaziheng/pretrain/internvl-sft/internvl_chat/shell/eval/evaluate_video4.sh"
SCRIPT6="/mnt/shared-storage-user/jiaziheng/pretrain/internvl-sft/internvl_chat/shell/eval/evaluate_video5.sh"
SCRIPT7="/mnt/shared-storage-user/jiaziheng/pretrain/internvl-sft/internvl_chat/shell/eval/evaluate_video6.sh"
SCRIPT8="/mnt/shared-storage-user/jiaziheng/pretrain/internvl-sft/internvl_chat/shell/eval/evaluate_video7.sh"


# 确保脚本有执行权限
chmod +x "$SCRIPT1" "$SCRIPT2" "$SCRIPT3" "$SCRIPT4" "$SCRIPT5" "$SCRIPT6" "$SCRIPT7" "$SCRIPT8"

# 同步（并行）运行多个脚本
wait
GPUS_DEVICE=$GPUS_DEVICE MODEL_PATH=$MODEL_PATH bash "$SCRIPT1" &
wait
GPUS_DEVICE=$GPUS_DEVICE MODEL_PATH=$MODEL_PATH bash "$SCRIPT2" &
wait
GPUS_DEVICE=$GPUS_DEVICE MODEL_PATH=$MODEL_PATH bash "$SCRIPT3" &
wait
GPUS_DEVICE=$GPUS_DEVICE MODEL_PATH=$MODEL_PATH bash "$SCRIPT4" &
wait
GPUS_DEVICE=$GPUS_DEVICE MODEL_PATH=$MODEL_PATH  bash "$SCRIPT5" &
wait
GPUS_DEVICE=$GPUS_DEVICE MODEL_PATH=$MODEL_PATH  bash "$SCRIPT6" &
wait
GPUS_DEVICE=$GPUS_DEVICE MODEL_PATH=$MODEL_PATH bash "$SCRIPT7" &
wait
GPUS_DEVICE=$GPUS_DEVICE MODEL_PATH=$MODEL_PATH bash "$SCRIPT8" &
wait



# 等待所有后台任务结束


echo "✅ 所有测试脚本已运行完成！"
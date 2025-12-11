#!/bin/bash
# 主脚本：同时运行多个子脚本

# 获取当前脚本所在目录（防止路径错误）
BASE_DIR=$(dirname "$0")

# 子脚本路径
SCRIPT1="/mnt/shared-storage-user/jiaziheng/pretrain/internvl-sft/internvl_chat/shell/eval/evaluate_image2.sh"
SCRIPT2="/mnt/shared-storage-user/jiaziheng/pretrain/internvl-sft/internvl_chat/shell/eval/evaluate_video4.sh"
# 确保脚本有执行权限
chmod +x "$SCRIPT1" "$SCRIPT2" "$SCRIPT3" "$SCRIPT4" "$SCRIPT5"

# 同步（并行）运行多个脚本
bash "$SCRIPT1" &
bash "$SCRIPT2" &
# bash "$SCRIPT3" &



# 等待所有后台任务结束
wait

echo "✅ 所有训练脚本已运行完成！"
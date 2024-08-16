#!/bin/bash

# 检查参数数量
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input_dir> <output_dir>"
  exit 1
fi

# 获取参数
input_dir="$1"
output_dir="$2"

# 创建输出目录（如果不存在）
mkdir -p "$output_dir"

# 运行 Python 脚本
python3 /workspace/detect.py "$input_dir" "$output_dir"
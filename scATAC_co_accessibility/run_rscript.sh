#!/bin/bash

dir="$1"
if [ -d "$dir" ]; then
    # 检查是否存在 all_peaks.csv 文件
    if [ -f "$dir/all_peaks.csv" ]; then
        echo "$dir 文件夹已经处理完成！"
    else
        echo "Processing directory: $dir"
        Rscript step1_preprocess_atac2peak.R "$dir"
    fi
fi
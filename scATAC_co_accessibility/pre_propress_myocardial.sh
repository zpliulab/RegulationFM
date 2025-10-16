#!/bin/bash


target_directory="/data/wcy_data/RegulationGPT_key/myocardial/new_preprocess"

find "$target_directory" -mindepth 3 -maxdepth 3  -type d -not -path "$target_directory" | parallel --jobs 16 ./run_rscript.sh {}

# 限制一级文件夹在指定列表，遍历三级目录中满足条件的子目录
#find "$target_directory" \
#    -mindepth 3 -maxdepth 3 -type d \
  #  -path "$target_directory/IZ_P3/*" -o \
  #  -path "$target_directory/GT_IZ_P9/*" -o \
  #  -path "$target_directory/IZ_P16/*" -o \
  #  -path "$target_directory/GT_IZ_P15/*" -o \
  #  -path "$target_directory/IZ_P15/*" -o \
  #  -path "$target_directory/GT_IZ_P13/*" | \
  #  grep -E '/(Myeloid|Fibroblast)$' | \
  #  parallel --jobs 16 ./run_rscript.sh {}

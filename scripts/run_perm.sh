#! /usr/bin/zsh

network=$1
dataset=$2
fast_wm=$3
method=$4

shift 4
files=("$@")

target_str=$(printf "'%s'," "${files[@]}")
target_str=${target_str%,}

target_arg="rebasin.targets=[$target_str]"

python3 ./srcs/find_permutations.py "$target_arg" \
    rebasin/method=$method \
    rebasin.fast_wm=$fast_wm \
    rebasin.network=$network \
    rebasin.dataset=$dataset

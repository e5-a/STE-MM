#!/usr/bin/zsh

num_models=10
# network = mlp or resnet20 or VGG
network=VGG
# dataset = MNIST or FMNIST or CIFAR10
dataset=CIFAR10
fast_wm=true
method=wm

for i in {1..$num_models}; do
    ./scripts/run_train.sh $network $dataset $i
done

models=`find ./outputs/train -name "*.ckpt" | tail -$num_models | tr "\n" " "`
./scripts/run_perm.sh $network $dataset $fast_wm $method `echo $models`

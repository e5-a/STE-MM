#!/usr/bin/zsh

num_models=4
# network = mlp or resnet20 or VGG
network=mlp
# dataset = MNIST or FMNIST or CIFAR10
dataset=MNIST
fast_wm=true
# method = ste or wm
method=ste

for i in {1..$num_models}; do
    ./scripts/run_train.sh $network $dataset $i
done

models=`find ./outputs/train -name "*.ckpt" | sort -n | tail -$num_models | tr "\n" " "`
./scripts/run_perm.sh $network $dataset $fast_wm $method `echo $models`

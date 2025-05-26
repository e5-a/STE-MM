#!/usr/bin/zsh


network=$1
dataset=$2
seed=$3

case "$network" in
  resnet20)
    epochs=100
    learning_rate=0.4
    batch_size=500
    #norm_layer='bn'
    weight_decay=0.0005
    train_optimizer='SGD'
    lr_scheduler='linear_up_down'
    momentum=0.9
    width=16
    ;;
  VGG)
    epochs=100
    learning_rate=0.4
    batch_size=500
    #norm_layer='bn'
    weight_decay=0.0005
    train_optimizer='SGD'
    lr_scheduler='linear_up_down'
    momentum=0.9
    width=4
    ;;
  mlp)
    epochs=5
    learning_rate=0.001
    batch_size=500
    #norm_layer='bn'
    weight_decay=0
    train_optimizer='Adam'
    lr_scheduler=''
    momentum=0.9
    width=1
    ;;
esac

python3 ./srcs/train.py train.epochs=$epochs \
    train.learning_rate=$learning_rate \
    train.batch_size=$batch_size \
    train.network.w=$width \
    train.seed=$seed \
    train.weight_decay=$weight_decay \
    train.optimizer=$train_optimizer \
    train.lr_scheduler=$lr_scheduler \
    train.momentum=$momentum \
    train/network=$network \
    train/dataset=$dataset

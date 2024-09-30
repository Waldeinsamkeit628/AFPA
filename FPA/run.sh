#!/bin/bash

cd ...

MODEL=fpa
DATA=cub
BACKBONE=resnet101
SAVE_PATH=.../${DATA}/.../${MODEL}

CUDA_VISIBLE_DEVICES=0 python main.py -a ${MODEL} -d ${DATA} -s ${SAVE_PATH} --backbone ${BACKBONE}  -b 128 --att 312 --lossw 10 --seed 6803 --phasew 0.2  --lr 0.02 --lr1 0.05 --epoch_decay 30 --pretrained --epochs 90 --is_fix
CUDA_VISIBLE_DEVICES=0  python main.py -a ${MODEL} -d ${DATA} -s ${SAVE_PATH} --backbone ${BACKBONE} -b 12 --att 312 --lossw 10 --seed 6803 --phasew 0.2  --lr 0.02 --lr1 0.001 --epoch_decay 30 --epochs 180 --resume ${SAVE_PATH}/fix.model




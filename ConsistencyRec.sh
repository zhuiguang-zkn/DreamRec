#!/bin/bash

# 设置超参数值
DATA="yc"
RANDOM_SEED=100
BATCH_SIZE=256
LAYERS=1
HIDDEN_FACTOR=64
LR=0.005
L2_DECAY=0
CUDA=1
DROPOUT_RATE=0.1
REPORT_EPOCH=true
DIFFUSER_TYPE="mlp1"
OPTIMIZER="adamw"
SIGMA_MIN=0.002
SIGMA_MAX=80.0
RHO=7.0
SIGMA_DATA=0.5
INITIAL_TIMESTEPS=2
FINAL_TIMESTEPS=150
LOSS_TYPE="l2"
TOTAL_TRAINING_STEP=300
EPOCH_EVERY_STEP=1
INTRODUCE_UNCONDITION=false

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --data)
            DATA="$2"
            shift
            shift
            ;;
        *)  # 其他未知参数忽略
            shift
            ;;
    esac
done

for LR in 0.00005 0.0001 0.001 0.01 ; do
    for EPOCH_EVERY_STEP in 1 2 3 4 5; do
        for FINAL_TIMESTEPS in 100 150 200 250; do 
            nohup python -u ConsistencyRec.py \
                --data $DATA \
                --random_seed $RANDOM_SEED \
                --batch_size $BATCH_SIZE \
                --layers $LAYERS \
                --hidden_factor $HIDDEN_FACTOR \
                --lr $LR \
                --l2_decay $L2_DECAY \
                --cuda $CUDA \
                --dropout_rate $DROPOUT_RATE \
                --report_epoch $REPORT_EPOCH \
                --diffuser_type $DIFFUSER_TYPE \
                --optimizer $OPTIMIZER \
                --descri "$DESCRI" \
                --sigma_min $SIGMA_MIN \
                --sigma_max $SIGMA_MAX \
                --rho $RHO \
                --sigma_data $SIGMA_DATA \
                --initial_timesteps $INITIAL_TIMESTEPS \
                --final_timesteps $FINAL_TIMESTEPS \
                --loss_type $LOSS_TYPE \
                --total_training_step $TOTAL_TRAINING_STEP \
                --epoch_every_step $EPOCH_EVERY_STEP \
                > ./log/ConsistencyRec_data_${DATA}_lr_${LR}_evs_${EPOCH_EVERY_STEP}_finalstep_${FINAL_TIMESTEPS}.log 2>&1 & 
        done
    done
done
                                                                            

#!/bin/bash

# To disable GPU, uncommit this
export CUDA_VISIBLE_DEVICES="-1"

# Avaliable dataset
declare -a datasets
datasets[0]='git-f6d5aac_jnetpcap-1.3.0'
datasets[1]='git-f6d5aac_jnetpcap-1.3.0_patch'
datasets[2]='git-f6d5aac_jnetpcap-1.4.r1425'
datasets[3]='git-f6d5aac_jnetpcap-1.4.r1425_patch'
datasets[4]='git-e3bb9ce_jnetpcap-1.3.0'
datasets[5]='git-e3bb9ce_jnetpcap-1.3.0_patch'
datasets[6]='git-e3bb9ce_jnetpcap-1.4.r1425'
datasets[7]='git-e3bb9ce_jnetpcap-1.4.r1425_patch'
datasets[8]='git-e3bb9ce_jnetpcap-1.4.r1500'
datasets[9]='git-e3bb9ce_jnetpcap-1.4.r1500_patch'
datasets[10]='git-98a5eba_jnetpcap-1.3.0'
datasets[11]='git-98a5eba_jnetpcap-1.4.r1425'

# Model
# 0:ANN  1:CNN  2:DNN  3:MLP  4:LSTM
models=(0 1 2 3 4)

# PCA rate
pca_rates=('0.85' '0.9' '0.95' '1' 'origin')

# rounds
rounds=3

# train.py
epochs=1000
batch_size=512


for dataset in "${datasets[@]}"
do
    for pca_rate in "${pca_rates[@]}"
    do
        python3 process_dataset.py $dataset $pca_rate
        for model_id in "${models[@]}"
        do
            mkdir -p "report/${dataset}/${pca_rate}/${model_id}"
            for round in $(seq 1 $rounds)
            do
                python3 train.py $model_id $epochs $batch_size
                mv report.csv "report/${dataset}/${pca_rate}/${model_id}/${round}.csv"
            done
        done
        rm tmp_data.npz
    done
done

#!/bin/bash
############################################################
# Load tmp_data.npz as data and train specified model.     #
# classification_report will save to 'report.csv'          #
#                                                          #
# Usage: main.py <model id>                                #
############################################################

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

# PCA rate
pca_rates=('0.85' '0.9' '0.95' '1' 'origin')

for dataset in "${datasets[@]}"
do
    for pca_rate in "${pca_rates[@]}"
    do
        python3 process_dataset.py $dataset $pca_rate
        for model_id in $(seq 0 3)
        do
            mkdir -p "report/${dataset}/${pca_rate}/${model_id}"
            for round in $(seq 0 2)
            do
                python3 train.py $model_id
                mv report.csv "report/${dataset}/${pca_rate}/${model_id}/${round}.csv"
            done
        done
        rm tmp_data.npz
    done
done

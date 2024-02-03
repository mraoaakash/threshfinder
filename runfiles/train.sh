#! /bin/bash
#PBS -N RUN_TRAIN
#PBS -o RUN_TRAIN_out.log
#PBS -e RUN_TRAIN_err.log
#PBS -l ncpus=10
#PBS -q gpu

module load compiler/anaconda3


eval "$(conda shell.bash hook)" 
conda activate threshfinder

for thresh in (1..100)
do
    for i in {"Xception","ResNet50","InceptionV3","VGG16","VGG19","MobileNetV2"}
    do
        for fold in {1..5}
        do
            python /home/aakash.rao_asp24/threshfinder/modelling/train.py \
                --base_path /home/aakash.rao_asp24/threshfinder \
                --model_name $i \
                --epochs 30 \
                --batch_size 8 \
                --lr 0.000001 \
                --threshold $ thresh \
                --fold $fold \
                # --mode train \

        done
    done
done
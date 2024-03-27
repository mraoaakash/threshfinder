#! /bin/bash
#PBS -N CellViT_Threshfinder
#PBS -o CellViT_Threshfinder_out.log
#PBS -e CellViT_Threshfinder_err.log
#PBS -l ncpus=50
#PBS -q gpu



INPRAWDIR=/storage/aakash.rao_asp24/research/research-MICCAI/input_raw
INDIR=/storage/aakash.rao_asp24/research/research-MICCAI/images
TEMPDIR=/storage/aakash.rao_asp24/research/research-MICCAI/temp
OUTDIR=/storage/aakash.rao_asp24/research/research-MICCAI/output



eval "$(conda shell.bash hook)"
conda activate Cellvit_env


# Move all files from raw to input
# find $INPRAWDIR -name '*.svs' | grep -v 'abdx' | xargs -I{} mv {} $INDIR
# find /storage/aakash.rao_asp24/research/research-MICCAI/input_raw/ -name '*.svs' | grep -v 'abdx' | xargs -I{} mv {} /storage/aakash.rao_asp24/research/research-MICCAI/images


for file in $INDIR/*.svs
do
    echo "Processing $file"
    file_path=$(basename $file)
    file_name=${file_path%.*}
    echo $file_name


    python3 /home/aakash.rao_asp24/thesis-supporters/CellViT/preprocessing/patch_extraction/main_extraction.py \
        --wsi_paths $file \
        --output_path $TEMPDIR \
        --patch_size 1024 \
        --patch_overlap 6.25 \

    
    # inference step
    python3 /home/aakash.rao_asp24/thesis-supporters/CellViT/cell_segmentation/inference/cell_detection.py \
        --model /home/aakash.rao_asp24/thesis-supporters/CellViT/models/pretrained/CellViT/CellViT-SAM-H-x40.pth \
        --batch_size 4 \
        --outdir_subdir $OUTDIR/$file_name \
        --geojson \
        process_wsi \
        --wsi_path $file \
        --patched_slide_path $TEMPDIR/$file_name \


    # clear temp
    rm -r $TEMPDIR/*
    # break
    
done
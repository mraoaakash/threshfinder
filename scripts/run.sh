INPRAWDIR=/mnt/storage/aakashrao/cifsShare/input_raw
INDIR=/mnt/storage/aakashrao/cifsShare/images
TEMPDIR=/mnt/storage/aakashrao/cifsShare/temp
OUTDIR=/mnt/storage/aakashrao/cifsShare/output



# Move all files from raw to input
# find $INPRAWDIR -name '*.svs' | grep -v 'abdx' | xargs -I{} mv {} $INDIR
# find /mnt/storage/aakashrao/cifsShare/input_raw/ -name '*.svs' | grep -v 'abdx' | xargs -I{} mv {} /mnt/storage/aakashrao/cifsShare/images


for file in $INDIR/*.svs
do
    echo "Processing $file"
    file_path=$(basename $file)
    file_name=${file_path%.*}
    echo $file_name


    python3 /home/aakashrao/research/research-MICCAI/CellViT/preprocessing/patch_extraction/main_extraction.py \
        --wsi_paths $file \
        --output_path $TEMPDIR \
        --patch_size 1024 \
        --patch_overlap 6.25 \

    
    # inference step
    python3 /home/aakashrao/research/research-MICCAI/CellViT/cell_segmentation/inference/cell_detection.py \
        --model /home/aakashrao/research/research-MICCAI/CellViT/models/pretrained/CellViT/CellViT-SAM-H-x40.pth \
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
#! /bin/bash
#PBS -N RUN_ORGANIZER
#PBS -o RUN_ORGANIZER_out.log
#PBS -e RUN_ORGANIZER_err.log
#PBS -l ncpus=10
#PBS -q gpu

module load compiler/anaconda3


eval "$(conda shell.bash hook)" 

python /home/aakash.rao_asp24/threshfinder/tools/organize_nucls.py \
    -i /home/aakash.rao_asp24/threshfinder/data/NuclsEvalSet


python /home/aakash.rao_asp24/threshfinder/tools/extract_patches.py \
    -b /home/aakash.rao_asp24/threshfinder



python /home/aakash.rao_asp24/threshfinder/tools/get_percs.py \
    -b /home/aakash.rao_asp24/threshfinder 
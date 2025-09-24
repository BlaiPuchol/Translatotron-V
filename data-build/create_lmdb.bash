#!/bin/bash
#SBATCH -p docencia             # cola
#SBATCH --gres=gpu:1            # nº de GPUs (máximo 4)
#SBATCH --cpus-per-task=16      # nº de CPUs (máximo 96)
#SBATCH --time=12:00:00         # tiempo máximo (opcional)
#SBATCH --job-name=test
#SBATCH -o salida_%j.log        # log de salida

export prefix="/home/alumno.upv.es/bpucsal/Translatotron-V"


export save_name=data-build/iwslt14.de-en-lmdb

python $prefix/data-build/create_lmdb_mulproc.py \
    --output_dir $prefix/$save_name \
    --text_data_dir $prefix/../datasets/IWSLT/iwslt14.de-en \
    --image_data_dir $prefix/../datasets/IWSLT/iwslt14.de-en-images \
    --src_lang de \
    --tgt_lang en \
    --num_workers 16

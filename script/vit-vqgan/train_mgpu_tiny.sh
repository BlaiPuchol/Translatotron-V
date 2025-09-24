#!/bin/bash
#SBATCH -p docencia             # cola
#SBATCH --gres=gpu:2            # nº de GPUs (máximo 4)
#SBATCH --cpus-per-task=16      # nº de CPUs (máximo 96)
#SBATCH --time=12:00:00         # tiempo máximo (opcional)
#SBATCH --job-name=test
#SBATCH -o salida_%j.log        # log de salida

export prefix="/home/alumno.upv.es/bpucsal/Translatotron-V"

vq_codebook_size=8192
vq_codebook_dim=64
dim=256
num_layers=4
batch_size=4
grad_accum_every=16
save_name="tiny-de-en"
image_dir="../datasets/IWSLT/iwslt14.de-en-images/train_de"

python -m torch.distributed.run --nproc_per_node=2 $prefix/src/train_mgpu.py \
    --output_dir $prefix/image-tokenizer/$save_name \
    --vq_codebook_size $vq_codebook_size \
    --vq_codebook_dim $vq_codebook_dim \
    --data_dir $prefix/$image_dir \
    --patch_size 16 \
    --dim $dim \
    --num_layers $num_layers \
    --batch_size $batch_size \
    --grad_accum_every $grad_accum_every

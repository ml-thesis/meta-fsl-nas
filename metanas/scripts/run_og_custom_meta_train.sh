#!/bin/bash

DATASET=omniglot
DATASET_DIR=/home/rob/Git/meta-fsl-nas/metanas/data
TRAIN_DIR=/home/rob/Git/meta-fsl-nas/metanas/custom/results
		
mkdir -p $TRAIN_DIR


args=(
    # Execution
    --name metatrain_og \
    --job_id 0 \
    --path ${TRAIN_DIR} \
    --data_path ${DATASET_DIR} \
    --dataset $DATASET
    --hp_setting 'og_metanas' \
    --use_hp_setting 1 \
    --workers 0 \
    --gpus 0 \
    --test_adapt_steps 1.0 \

    # few shot params
     # examples per class
    --n 5 \
    # number classes  
    --k 20 \
    # test examples per class
    --q 1 \

    --meta_model_prune_threshold 0.01 \
    --alpha_prune_threshold 0.01 \
    # Meta Learning
    # Original settings 30_000 meta epochs
    # and warm_up_epochs 15_000.
    --meta_model searchcnn \
    --meta_epochs 75 \
    --warm_up_epochs 10 \
    --use_pairwise_input_alphas \
    # --eval_freq 2500 \
    --eval_freq 2 \
    # --eval_epochs 200 \
    --eval_epochs 2 \

    --normalizer softmax \
    --normalizer_temp_anneal_mode linear \
    --normalizer_t_min 0.05 \
    --normalizer_t_max 1.0 \
    --drop_path_prob 0.2 \

    # Architectures
    --init_channels 28 \
    --layers 5 \
    --reduction_layers 2 4 \
    --use_first_order_darts \

    --use_torchmeta_loader \

)


python -u -m metanas.metanas_main "${args[@]}"
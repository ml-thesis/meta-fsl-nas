#!/bin/bash

DATASET=omniglot
DATASET_DIR=/home/rob/Git/meta-fsl-nas/metanas/data
TRAIN_DIR=/home/rob/Git/meta-fsl-nas/metanas/results/og_train
		
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

    --meta_model_prune_threshold 0.001 \
    --alpha_prune_threshold 0.001 \
    # Meta Learning
    # Original settings 30_000 meta epochs
    # warm_up_epochs 15_000
    --meta_model searchcnn \
    --meta_epochs 18 \
    --warm_up_epochs 3 \
    --use_pairwise_input_alphas \
    # --eval_freq 2500 \
    --eval_freq 5 \
    # --eval_epochs 200 \
    --eval_epochs 3 \

    --normalizer softmax \
    --normalizer_temp_anneal_mode linear \
    --normalizer_t_min 0.05 \
    --normalizer_t_max 1.0 \
    # In P-DARTS 0.3 is used in metaNAS 0.2
    --drop_path_prob 0.3 \

    # Architectures
    --init_channels 28 \
    --layers 4 \
    --nodes 3 \
    --reduction_layers 1 3 \
    --use_first_order_darts \

    --use_torchmeta_loader \

)


python -u -m metanas.metanas_main "${args[@]}"
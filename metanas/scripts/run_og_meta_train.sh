#!/bin/bash

DATASET=omniglot
DATASET_DIR=/home/rob/Git/meta-fsl-nas/data
TRAIN_DIR=/home/rob/Git/meta-fsl-nas/metanas/results/og_train_v4
		
mkdir -p $TRAIN_DIR


args=(
    # Execution
    --name metatrain_og_v4 \
    --job_id 0 \
    --path ${TRAIN_DIR} \
    --data_path ${DATASET_DIR} \
    --dataset $DATASET
    --hp_setting 'og_pdarts' \
    --use_hp_setting 1 \
    --workers 0 \
    --gpus 0 \
    --test_adapt_steps 1.0 \

    # few shot params
     # examples per class
    --n 1 \
    # number classes
    --k 20 \
    # test examples per class
    --q 1 \

    # Originally, 0.01 for 30_000 epochs
    --meta_model_prune_threshold 0.01 \
    --alpha_prune_threshold 0.01 \
    # Meta Learning
    --meta_model searchcnn \
    # Repeated every stage, e.g. 3 times
    --meta_epochs 25 \
    --warm_up_epochs 10 \
    --use_pairwise_input_alphas \
    --eval_freq 15 \
    --eval_epochs 5 \

    --print_freq 5 \

    --normalizer softmax \
    --normalizer_temp_anneal_mode linear \
    --normalizer_t_min 0.05 \
    --normalizer_t_max 1.0 \
    # Originally, 0.2 in MetaNAS
    # Test this setting
    --drop_path_prob 0.3 \

    # Architectures
    --init_channels 28 \
    # P-DARTS layers, reduction layers and nodes
    --layers 5 \
    --nodes 4 \
    --reduction_layers 2 4 \
    --use_first_order_darts \
    --use_torchmeta_loader \

    # P-DARTS
    --use_search_space_approximation \
    --use_search_space_regularization \

    # Add the same weight decay as PDARTS
    --w_weight_decay 0.0003
    --alpha_weight_decay 0.001

    --add_layers 5 \
    --add_init_channels 12 \
    --limit_skip_connections 2 \
)


python -u -m metanas.metanas_main "${args[@]}"
#!/bin/bash

DATASET=omniglot
DATASET_DIR=/home/rob/Git/meta-fsl-nas/data
TRAIN_DIR=/home/rob/Git/meta-fsl-nas/metanas/results/test_exp
		
mkdir -p $TRAIN_DIR


args=(
    # Execution
    --name test_exp \
    --job_id 0 \
    --path ${TRAIN_DIR} \
    --data_path ${DATASET_DIR} \
    --dataset $DATASET
    --hp_setting 'test_exp' \
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

    --meta_model_prune_threshold 0.001 \
    --alpha_prune_threshold 0.001 \
    # Meta Learning
    --meta_model searchcnn \
    --meta_epochs 2 \
    --test_task_train_steps 2 \

    --warm_up_epochs 1 \
    --use_pairwise_input_alphas \
    --eval_freq 2 \
    --eval_epochs 1 \
    --print_freq 2 \

    --normalizer softmax \
    --normalizer_temp_anneal_mode linear \
    --normalizer_t_min 0.05 \
    --normalizer_t_max 1.0 \
    --drop_path_prob 0.2 \

    # Architectures
    --init_channels 28 \
    --layers 4 \
    --nodes 3 \
    --reduction_layers 1 3 \
    --use_first_order_darts \
    --use_torchmeta_loader \

    # P-DARTS
    --use_search_space_approximation \
    --use_search_space_regularization \

)


python -u -m metanas.metanas_main "${args[@]}"
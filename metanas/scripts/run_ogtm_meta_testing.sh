#!/bin/bash

DATASET=mixedomniglottriplemnist
DATASET_DIR=/home/rob/Git/meta-fsl-nas/metanas/data
TRAIN_DIR=/home/rob/Git/meta-fsl-nas/metanas/results/ogtm_test
		
mkdir -p $TRAIN_DIR

# TODO: First run the ogtm meta train
# MODEL_PATH=/path/to/checkpoint/from/metatrain


args=(
    # Execution
    --name metatest_ogtm \
    --job_id 0 \
    --path ${TRAIN_DIR} \
    --data_path ${DATASET_DIR} \
    --dataset $DATASET
    --hp_setting 'ogtm_metanas' \
    --use_hp_setting 1 \
    --workers 0 \
    --gpus 0 \
    --test_adapt_steps 0.5 \
    --test_task_train_steps 100 \
    --eval \
    # few shot params
     # examples per class
    --n 1 \
    # number classes  
    --k 20 \
    # test examples per class
    --q 1 \

    --meta_model_prune_threshold 0.01 \
    --alpha_prune_threshold 0.05 \
    # Meta Learning
    --meta_model searchcnn \
    --model_path ${MODEL_PATH}
    --meta_epochs 1000 \
    --warm_up_epochs 500 \
    --use_pairwise_input_alphas \
    --eval_freq 200 \
    --eval_epochs 20 \

    # Ablation study
    --use_search_space_approximation
    --use_search_space_regularization

    --normalizer softmax \
    --normalizer_temp_anneal_mode linear \
    --normalizer_t_min 0.05 \
    --normalizer_t_max 1.0 \
    --drop_path_prob 0.2 \

    # Architectures
    --init_channels 28 \
    --layers 4 \
    --reduction_layers 1 3 \
    --use_first_order_darts \

    --use_torchmeta_loader \

)


python -u -m metanas.metanas_main "${args[@]}"
#!/bin/bash

DATASET=triplemnist
DATASET_DIR=/home/rob/Git/meta-fsl-nas/data
TRAIN_DIR=/home/rob/Git/meta-fsl-nas/metanas/results/transfer/og_tm_train
		
mkdir -p $TRAIN_DIR

MODEL_PATH=/home/rob/Git/meta-fsl-nas/metanas/results/ablation/og_train_metanas/meta_state

args=(
    # Execution
    --name metatrain_og \
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

    --meta_model_prune_threshold 0.01 \
    --alpha_prune_threshold 0.01 \
    # Meta Learning
    --meta_model searchcnn \
    --model_path ${MODEL_PATH} \
    --meta_epochs 75 \
    --warm_up_epochs 35 \
    --use_pairwise_input_alphas \

    --eval_freq 15 \
    --eval_epochs 5 \
    --print_freq 5 \

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

    # P-DARTS & sharpDARTS
    # TODO: To be decided what is the best combination.
)


python -u -m metanas.metanas_main "${args[@]}"
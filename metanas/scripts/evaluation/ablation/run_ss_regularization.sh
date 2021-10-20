#!/bin/bash

DATASET=$DS
DATASET_DIR=/home/rob/Git/meta-fsl-nas/data

for SEED in ${SEEDS}
do
    TRAIN_DIR=/home/rob/Git/meta-fsl-nas/metanas/results/ablation/${DS}_train_ssr/${DS}_train_ssr_$SEED
	mkdir -p $TRAIN_DIR

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

        --seed $SEED
        
        # few shot params
        # examples per class
        --n $N \
        # number classes
        --k 20 \
        # test examples per class
        --q 1 \

        --meta_model_prune_threshold 0.01 \
        --alpha_prune_threshold 0.01 \
        # Meta Learning
        --meta_model searchcnn \
        --meta_epochs $EPOCHS \
        --warm_up_epochs $WARM_UP \
        --use_pairwise_input_alphas \

        --eval_freq $EVAL_FREQ \
        --eval_epochs 50 \
        --print_freq 100 \

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
        --use_search_space_regularization \
        --limit_skip_connections 2 \
    )

    python -u -m metanas.metanas_main "${args[@]}"

done
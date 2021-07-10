#!/bin/bash

DATASET_1=omniglot
DATASET_2=triplemnist
DATASET_DIR=/home/rob/Git/meta-fsl-nas/data
TRAIN_DIR=/home/rob/Git/meta-fsl-nas/metanas/results/og_transfer
		
mkdir -p $TRAIN_DIR

MODEL_PATH=/home/rob/Git/meta-fsl-nas/metanas/results/og_transfer

args_omni=(
    # Execution
    --name metatrain_og \
    --job_id 0 \
    --path ${TRAIN_DIR} \
    --data_path ${DATASET_DIR} \
    --dataset $DATASET_1
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
    --meta_model_prune_threshold 0.001 \
    --alpha_prune_threshold 0.001 \
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
    # Originally, 0.3 in PDARTS
    # Test this setting
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

    --add_layers 5 \
    --add_init_channels 12 \
    --limit_skip_connections 2 \
)

# First meta-train on omniglot
python -u -m metanas.metanas_main "${args_omni[@]}"


# Second, meta-train on tripleMNIST to test transfer
args_tm=(
    # Execution
    --name metatest_og \
    --job_id 0 \
    --path ${TRAIN_DIR} \
    --data_path ${DATASET_DIR} \
    --dataset $DATASET_2
    --hp_setting 'og_pdarts' \
    --use_hp_setting 1 \
    --workers 0 \
    --gpus 0 \
    --test_adapt_steps 0.5 \

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
    --model_path ${MODEL_PATH}
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
    # Originally, 0.3 in metaNAS
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

    --add_layers 5 \
    --add_init_channels 12 \
    --limit_skip_connections 2 \
)

python -u -m metanas.metanas_main "${args_tm[@]}"
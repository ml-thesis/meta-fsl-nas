#!/bin/bash

# parameters
# TODO: Add eval_freq and n
EPOCHS=15
WARM_UP_EPOCHS=5
SEEDS=(0 1 2)
DS=omniglot

echo "Start run of ablation studies, variables epochs = ${EPOCHS}, warm up variables = ${WARM_UP_EPOCHS}, seeds = ${SEEDS[@]}, dataset = ${DS}"

# MetaNAS baseline
EPOCHS=$EPOCHS WARM_UP=$WARM_UP_EPOCHS SEEDS=${SEEDS[@]} DS=$DS ./scripts/ablation/run_metanas_baseline.sh

# SharpDARTS
EPOCHS=$EPOCHS WARM_UP=$WARM_UP_EPOCHS SEEDS=${SEEDS[@]} DS=$DS ./scripts/ablation/run_cosine_power_annealing.sh
EPOCHS=$EPOCHS WARM_UP=$WARM_UP_EPOCHS SEEDS=${SEEDS[@]} DS=$DS ./scripts/ablation/run_alpha_regularization.sh
EPOCHS=$EPOCHS WARM_UP=$WARM_UP_EPOCHS SEEDS=${SEEDS[@]} DS=$DS ./scripts/ablation/run_ss_sharpdarts.sh

# P-DARTS
EPOCHS=$EPOCHS WARM_UP=$WARM_UP_EPOCHS SEEDS=${SEEDS[@]} DS=$DS ./scripts/ablation/run_ss_approximation.sh
EPOCHS=$EPOCHS WARM_UP=$WARM_UP_EPOCHS SEEDS=${SEEDS[@]} DS=$DS ./scripts/ablation/run_ss_regularization.sh
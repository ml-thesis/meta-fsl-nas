#!/bin/bash

# parameters
EPOCHS=500
WARM_UP_EPOCHS=250
SEEDS=(0) # 1 2
EVAL_FREQ=100
N=1 # 3, 5
DS=omniglot

echo "Start run of ablation studies, variables epochs = ${EPOCHS}, warm up variables = ${WARM_UP_EPOCHS}, seeds = ${SEEDS[@]}, dataset = ${DS}"

# MetaNAS baseline
# EPOCHS=$EPOCHS WARM_UP=$WARM_UP_EPOCHS SEEDS=${SEEDS[@]} DS=$DS N=$N EVAL_FREQ=$EVAL_FREQ ./scripts/ablation/run_metanas_baseline.sh

# SharpDARTS
EPOCHS=$EPOCHS WARM_UP=$WARM_UP_EPOCHS SEEDS=${SEEDS[@]} DS=$DS N=$N EVAL_FREQ=$EVAL_FREQ ./scripts/ablation/run_alpha_regularization.sh
EPOCHS=$EPOCHS WARM_UP=$WARM_UP_EPOCHS SEEDS=${SEEDS[@]} DS=$DS N=$N EVAL_FREQ=$EVAL_FREQ ./scripts/ablation/run_ss_sharpdarts.sh
# EPOCHS=$EPOCHS WARM_UP=$WARM_UP_EPOCHS SEEDS=${SEEDS[@]} DS=$DS N=$N EVAL_FREQ=$EVAL_FREQ ./scripts/ablation/run_cosine_power_annealing.sh

# P-DARTS
# EPOCHS=$EPOCHS WARM_UP=$WARM_UP_EPOCHS SEEDS=${SEEDS[@]} DS=$DS N=$N EVAL_FREQ=$EVAL_FREQ ./scripts/ablation/run_ss_approximation.sh
# EPOCHS=$EPOCHS WARM_UP=$WARM_UP_EPOCHS SEEDS=${SEEDS[@]} DS=$DS N=$N EVAL_FREQ=$EVAL_FREQ ./scripts/ablation/run_ss_regularization.sh

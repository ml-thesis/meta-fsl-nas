#!/bin/bash

source /home/TUE/20184291/miniconda3/etc/profile.d/conda.sh
source activate metanas

# parameters
EPOCHS=500
WARM_UP_EPOCHS=250
SEEDS=(2)
SEEDS_TWO=(1 2)
EVAL_FREQ=100
N=1
DS=omniglot

echo "Start run of ablation studies, variables epochs = ${EPOCHS}, warm up variables = ${WARM_UP_EPOCHS}, seeds = ${SEEDS[@]}, dataset = ${DS}"

# MetaNAS baseline
# EPOCHS=$EPOCHS WARM_UP=$WARM_UP_EPOCHS SEEDS=${SEEDS[@]} DS=$DS N=$N EVAL_FREQ=$EVAL_FREQ ./scripts/ablation/run_metanas_baseline.sh

# SharpDARTS
# EPOCHS=$EPOCHS WARM_UP=$WARM_UP_EPOCHS SEEDS=${SEEDS[@]} DS=$DS N=$N EVAL_FREQ=$EVAL_FREQ ./scripts/ablation/run_alpha_regularization.sh
# EPOCHS=$EPOCHS WARM_UP=$WARM_UP_EPOCHS SEEDS=${SEEDS[@]} DS=$DS N=$N EVAL_FREQ=$EVAL_FREQ ./scripts/ablation/run_ss_sharpdarts.sh
# EPOCHS=$EPOCHS WARM_UP=$WARM_UP_EPOCHS SEEDS=${SEEDS[@]} DS=$DS N=$N EVAL_FREQ=$EVAL_FREQ ./scripts/ablation/run_cosine_power_annealing.sh

# P-DARTS
# EPOCHS=$EPOCHS WARM_UP=$WARM_UP_EPOCHS SEEDS=${SEEDS[@]} DS=$DS N=$N EVAL_FREQ=$EVAL_FREQ ./scripts/ablation/run_ss_approximation.sh
# EPOCHS=$EPOCHS WARM_UP=$WARM_UP_EPOCHS SEEDS=${SEEDS[@]} DS=$DS N=$N EVAL_FREQ=$EVAL_FREQ ./scripts/ablation/run_ss_regularization.sh

# Custom adjustments
# TODO: Still need to run on seed 1
EPOCHS=$EPOCHS WARM_UP=$WARM_UP_EPOCHS SEEDS=${SEEDS_TWO[@]} DS=$DS N=$N EVAL_FREQ=$EVAL_FREQ ./scripts/ablation/run_dropout_and_limit_sk.sh
EPOCHS=$EPOCHS WARM_UP=$WARM_UP_EPOCHS SEEDS=${SEEDS[@]} DS=$DS N=$N EVAL_FREQ=$EVAL_FREQ ./scripts/ablation/run_dropout_skip_conn.sh
EPOCHS=$EPOCHS WARM_UP=$WARM_UP_EPOCHS SEEDS=${SEEDS[@]} DS=$DS N=$N EVAL_FREQ=$EVAL_FREQ ./scripts/ablation/run_limit_sk.sh
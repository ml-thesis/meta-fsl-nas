#!/bin/bash

# source /home/TUE/20184291/miniconda3/etc/profile.d/conda.sh
# source activate metanas

# TODO: Start tensorboard

# parameters
EPOCHS=250
WARM_UP_EPOCHS=125
SEEDS=(2)
SEEDS_TWO=(1)
EVAL_FREQ=75
N=1
DS=omniglot

echo "Start run of ablation studies, variables epochs = ${EPOCHS}, warm up variables = ${WARM_UP_EPOCHS}, seeds = ${SEEDS[@]}, dataset = ${DS}"

# MetaNAS baseline
EPOCHS=$EPOCHS WARM_UP=$WARM_UP_EPOCHS SEEDS=${SEEDS[@]} DS=$DS N=$N EVAL_FREQ=$EVAL_FREQ ./scripts/agent/run_agent_baseline.sh

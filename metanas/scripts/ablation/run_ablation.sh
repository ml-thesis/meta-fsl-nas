#!/bin/bash

echo "Start run of ablation studies omitting, search space approximation and regularization"
sh ./run_og_ssa_meta_train.sh

sh ./run_og_ssr_meta_train.sh


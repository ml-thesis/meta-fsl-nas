#!/bin/bash

echo "Start run of ablation studies omitting, search space approximation and regularization"

# MetaNAS
# bash ../../metanas/scripts/run_og_meta_train.sh
# bash ./scripts/ablation/run_metanas.sh
# bash ./scripts/ablation/run_cutout.sh

# SharpDARTS
# bash ./scripts/ablation/run_sharp.sh
# bash ./scripts/ablation/run_alpha_reg.sh
# bash ./scripts/ablation/run_ss_sharp.sh

# P-DARTS
# bash ./scripts/ablation/run_reinitialize_ssa.sh
bash ./scripts/ablation/run_pdarts.sh
bash ./scripts/ablation/run_ssr.sh
bash ./scripts/ablation/run_ssa.sh
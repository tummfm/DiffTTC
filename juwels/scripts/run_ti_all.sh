#! /bin/bash -x
# Setup list of parameters to loop over.

#SBATCH --account=mfm
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/solidfe-bcc-hcp-gpu-out.%j
#SBATCH --error=logs/solidfe-bcc-hcp-gpu-err.%j
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12

bash scripts/run_ti_bcc.sh &
bash scripts/run_ti_hcp.sh &

wait

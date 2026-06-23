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

export OMP_NUM_THREADS=12

tag="refined"

## Pretained model
if [ "$tag" == "pretrained" ]; then
  model="titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5"

elif [ "$tag" == "refined" ]; then
  model="titanium_train_solid_MACE_r_cutoff_0.5_2025_11_13_f15304d6-4928-486b-b4e7-1fa652dadbff"

fi

# Run at experimental temperature
T=("1940.52" "1979.41" "2011.3" "2041.43" "2071.56" "2092.93")
P=(0 1 2 3 4 5)

# Run job.
for n in "${!T[@]}"
do
  folderpath="${SCRATCH}/projects/crystal_fem/bcc-liquid/${model}/prepare_T_${T[n]}_P_${P[n]}/"
  mkdir -p $folderpath # Create directory structure for data output.

  RANDOM="13244" # Generate a random number for each run.

  # Create and equilibrate bcc structure.
  echo "Run BCC Preparation"
  srun --nodes 1 --exclusive -n 1 --gres=gpu:1 \
  lmp       -in lammps/prepare_bcc.lmp  \
            -log ${folderpath}/prepare_bcc.log \
            -var model "$model"          \
            -var RANDOM ${RANDOM}         \
            -var EstTemp ${T[n]}          \
            -var folderpath ${folderpath} \
            -var DefPress ${P[n]}         \
            -var structure bcc &

    # Create and equilibrate bcc structure.
  echo "Run Liquid Preparation"
  srun --nodes 1 --exclusive -n 1 --gres=gpu:1 \
  lmp       -in lammps/prepare_liquid.lmp  \
            -log ${folderpath}/prepare_liquid.log \
            -var model "$model"          \
            -var RANDOM ${RANDOM}         \
            -var EstTemp ${T[n]}          \
            -var folderpath ${folderpath} \
            -var DefPress ${P[n]}         \
            -var structure liquid &

done

wait
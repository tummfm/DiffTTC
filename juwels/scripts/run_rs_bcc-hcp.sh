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

## Pretrained ##############################################################
if [ "$tag" == "pretrained" ]; then
  model="titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5"
  # Run RS at experimental temperature
  T=("1154.7" "1124.5" "1094.4" "1064.3" "1034.2" "1005.0")
  P=(0 1 2 3 4 5)

  # Run towards the coexistence temperature
  Ts=("800" "770" "740" "710" "700" "700")

elif [ "$tag" == "refined" ]; then
  model="titanium_train_solid_MACE_r_cutoff_0.5_2025_11_13_f15304d6-4928-486b-b4e7-1fa652dadbff"

  # Start at experimental temperature
  T=("1154.7" "1124.5" "1094.4" "1064.3" "1034.2" "1005.0")
  P=(0 1 2 3 4 5)

  Ts=("1000" "1030" "1040" "1090" "1100" "1120")

fi


# Run job.
for struct in "bcc" "hcp"; do
  if [ "$struct" == "bcc" ]; then
    echo "Running BCC reversible scaling"
    couple="xyz"
  else
    echo "Running HCP reversible scaling"
    couple="xy"
  fi

  for n in "${!T[@]}"; do
    folderpath="${SCRATCH}/projects/crystal_fem/solidfe/hcp-bcc/${model}/reversible_scaling_T_${T[n]}_P_${P[n]}_${struct}/"
    prepared="${SCRATCH}/projects/crystal_fem/solidfe/hcp-bcc/${model}/prepare_T_${T[n]}_P_${P[n]}/${struct}_equilibrated_P_${P[n]}_T_${T[n]}.lmpdat"

    if [ -d "$folderpath" ]; then
      echo "Folder already exists, skipping: $folderpath"
      continue
    fi

    if [ ! -f "$prepared" ]; then
      echo "Prepared file not found: $prepared"
      exit 1
    fi

    mkdir -p $folderpath # Create directory structure for data output.

    RANDOM="13244" # Generate a random number for each run.

    # Create and equilibrate bcc structure.
    srun --nodes 1 --exclusive -n 1 --gres=gpu:1 \
    lmp       -in lammps/reversible_scaling.lmp  \
              -log ${folderpath}/reversible_scaling.log \
              -var folderpath ${folderpath} \
              -var model "$model"          \
              -var RANDOM ${RANDOM}         \
              -var T ${T[n]}          \
              -var P ${P[n]}         \
              -var Ts ${Ts[n]}       \
              -var couple $couple       \
              -var prepared $prepared &
  done
done

wait

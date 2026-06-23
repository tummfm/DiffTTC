#! /bin/bash -x
# Setup list of parameters to loop over.

#SBATCH --account=mfm
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/solidfe-bcc-liquid-gpu-out.%j
#SBATCH --error=logs/solidfe-bcc-liquid-gpu-err.%j
#SBATCH --time=10:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12

export OMP_NUM_THREADS=12

tag="refined"

## Pretrained ##############################################################
if [ "$tag" == "pretrained" ]; then
  model="titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5"
  # Run RS at experimental temperature
  T=("1940.52" "1979.41" "2011.3"  "2041.43" "2071.56" "2092.93")
  P=(0 1 2 3 4 5)

  # Run towards the coexistence temperature
  input="${SCRATCH}/projects/crystal_fem/coexistence/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/temps.txt"

elif [ "$tag" == "refined" ]; then
  model="titanium_train_solid_MACE_r_cutoff_0.5_2025_11_13_f15304d6-4928-486b-b4e7-1fa652dadbff"

  # Start at experimental temperature
  T=("1940.52" "1979.41" "2011.3"  "2041.43" "2071.56" "2092.93")
  P=(0 1 2 3 4 5)

  input="${SCRATCH}/projects/crystal_fem/coexistence/titanium_train_solid_MACE_r_cutoff_0.5_2025_11_13_f15304d6-4928-486b-b4e7-1fa652dadbff/temps.txt"

fi

declare -a Ts

if [ ! -f "$input" ]; then
  echo "File $input not found. Exiting..."
  exit 1
fi

skip_header=true
while read -a cols; do
  if [ "$skip_header" = true ]; then
    skip_header=false
    continue
  fi
    echo "Run: ${cols[0]} GPa | Ts = ${cols[2]} K"
    Ts+=("${cols[2]}")
done < $input

# Run job.
for struct in "bcc" "liquid"; do

  for n in "${!T[@]}"; do
    folderpath="${SCRATCH}/projects/crystal_fem/bcc-liquid/${model}/reversible_scaling_T_${T[n]}_P_${P[n]}_${struct}_Ts_${Ts[n]}/"
    prepared="${SCRATCH}/projects/crystal_fem/bcc-liquid/${model}/prepare_T_${T[n]}_P_${P[n]}/${struct}_equilibrated_P_${P[n]}_T_${T[n]}.lmpdat"

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
              -var couple "xyz"       \
              -var prepared $prepared &
  done
done

wait


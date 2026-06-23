#! /bin/bash -x

#SBATCH --account=mfm
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/gpu-out.%j
#SBATCH --error=logs/gpu-err.%j
#SBATCH --time=10:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4

export OMP_NUM_THREADS=12

## Pretrained model ####################################

tag="refined"

if [ "$tag" == "pretrained" ]; then
  model="titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5"

  # Initial guesses for coexistence temperatures at different pressures
  temps=("1743" "1775" "1804" "1831" "1856" "1878")

elif [ "$tag" == "refined" ]; then
  model="titanium_train_solid_MACE_r_cutoff_0.5_2025_11_13_f15304d6-4928-486b-b4e7-1fa652dadbff"

  temps=("1940.52" "1979.41" "2011.3" "2041.43" "2071.56" "2092.93")

fi

## Run at experimental temperature
press=(0 1 2 3 4 5)

# Rerun the coexistence method from a slightly higher or lower temperature
delta=(0)

# Loop over the temperatures and pressures and run coexistence method
for j in "${!delta[@]}"; do
    echo "Delta ${delta[$j]}"
    for i in "${!temps[@]}"; do
        pressure=${press[$i]}
        now=$(date +%Y%m%d%H%M)

        folderpath="${SCRATCH}/projects/crystal_fem/coexistence/${model}/coexistence_T_${temps[$i]}_P_${press[$i]}_delta_${delta[$j]}_${now}/"
        mkdir -p "$folderpath"

        echo "Running coexistence for temperature ${temps[$i]} and pressure ${pressure}"

        srun --nodes 1 --exclusive -n 2 --gres=gpu:2 \
        lmp -in lammps/coexistence.lmp \
             -log "$folderpath/coexistence.log" \
             -v DateTime "$now" \
             -v EstTemp "${temps[$i]}" \
             -v DefPress "$pressure" \
             -v model "$model" \
             -v delta "${delta[$j]}" \
             -v folderpath "$folderpath" &

    done
done

# Wait for all background jobs to finish
wait

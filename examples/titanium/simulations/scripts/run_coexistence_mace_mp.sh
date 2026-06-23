#! /bin/bash

export OMP_NUM_THREADS=12

model="mace-mp-0b3-medium.model-lammps.pt"

temps=("1450" "1490" "1520" "1540" "1560" "1580")
press=(0 1 2 3 4 5)

delta=(0)

# Loop over the temperatures and pressures and run coexistence method
for j in "${!delta[@]}"; do
  for i in "${!temps[@]}"; do
      now=$(date +%Y%m%d%H%M)

      folderpath="coexistence/output/${model}/coexistence_T_${temps[$i]}_P_${press[$i]}_delta_${delta[$j]}/"

      if [ -d "$folderpath" ]; then
        echo "Folder already exists, skipping: $folderpath"
        continue
      fi

      mkdir -p "$folderpath"
      echo "Running coexistence for temperature ${temps[$i]} and pressure ${press[$i]}"

      lmpkk -k on g 1 -sf kk -pk kokkos newton on neigh half \
            -in lammps/coexistence_mace_mp.lmp \
            -log "$folderpath/coexistence.log" \
            -v DateTime "$now" \
            -v EstTemp "${temps[$i]}" \
            -v DefPress "${press[$i]}" \
            -v model "$model" \
            -v delta "${delta[$j]}" \
            -v folderpath "$folderpath"

    done
done

#! /bin/bash

## Pretrained model ####################################

# Estimated melting temperatures based on DP potential and corresponding pressures
# temps=(1880 1900 1930 1970 1980 2000)
# temps=(1720 1770 1800 1840 1860 1880)
# temps=(1941 1979 2011 2041 2072 2093)
# press=(0 1 2 3 4 5)
# temps=(1720)
# press=(0)

tag="refined"

if [ "$tag" == "pretrained" ]; then

  model="titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5"

  press=(0 1 2 3 4 5)
  temps=(1940 1980 2010 2040 2070 2090)

## Model trained on all transitions + higher lr + higher pressure coef + simultaneous pair selection ###
elif [ "$tag" == "refined" ]; then
  model="titanium_train_solid_MACE_r_cutoff_0.5_2025_11_13_f15304d6-4928-486b-b4e7-1fa652dadbff"

  press=(0 1 2 3 4 5)
  temps=("1940.52" "1979.41" "2011.3" "2041.43" "2071.56" "2092.93")

fi

## Run at experimental temperature
# temps=("1940.52" "1979.41" "2011.3")

# Rerun the coexistence method from a slightly higher or lower temperature
delta=(0 -10 +10)

export OMP_NUM_THREADS=$(nproc --all)

# Loop over the temperatures and pressures and run coexistence method
for j in "${!delta[@]}"; do
    echo "Delta ${delta[$j]}"
    for i in "${!temps[@]}"; do
        pressure=${press[$i]}
        now=$(date +%Y%m%d%H%M)

        folderpath="./coexistence/output/${model}/coexistence_T_${temps[$i]}_P_${press[$i]}_delta_${delta[$j]}_${now}/"
        mkdir -p "$folderpath"

        echo "Running coexistence for temperature ${temps[$i]} and pressure ${pressure}"
        lmp -in lammps/coexistence.lmp \
               -log "$folderpath/coexistence.log" \
               -v DateTime "$now" \
               -v EstTemp "${temps[$i]}" \
               -v DefPress "$pressure" \
               -v model "$model" \
               -v delta "${delta[$j]}" \
               -v folderpath "$folderpath" 



    done
done


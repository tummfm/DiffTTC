# Setup list of parameters to loop over.

export OMP_NUM_THREADS=8

tag="pretrained"

## Pretained model
if [ "$tag" == "pretrained" ]; then
  model="titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5"

elif [ "$tag" == "lammps" ]; then
  model="titanium_train_solid_MACE_r_cutoff_0.5_2025_10_25_d1d7ab61-d1dd-4862-a944-3d7622576ab0"

fi

## Run at experimental temperature
T=("1940.52" "1979.41" "2011.3" "2041.43" "2071.56" "2092.93")
P=(0 1 2 3 4 5)

# Run job.
for n in "${!T[@]}"
do
  folderpath="./solidfe/output/bcc-liquid/${model}/melting/prepare_T_${T[n]}_P_${P[n]}/"
  mkdir -p $folderpath # Create directory structure for data output.

  RANDOM="13244" # Generate a random number for each run.

  # Create and equilibrate liquid structure.
  lmp       -in lammps/prepare_liquid.lmp  \
            -log ${folderpath}/prepare_liquid.log \
            -var model "$model"          \
            -var RANDOM ${RANDOM}         \
            -var EstTemp ${T[n]}          \
            -var folderpath ${folderpath} \
            -var DefPress ${P[n]}         \
            -var structure liquid

  # Create and equilibrate bcc structure.
  echo "Run BCC Preparation"
  lmp       -in lammps/prepare_bcc.lmp  \
            -log ${folderpath}/prepare_bcc.log \
            -var model "$model"          \
            -var RANDOM ${RANDOM}         \
            -var EstTemp ${T[n]}          \
            -var folderpath ${folderpath} \
            -var DefPress ${P[n]}         \
            -var structure bcc

done


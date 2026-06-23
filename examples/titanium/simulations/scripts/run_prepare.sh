# Setup list of parameters to loop over.

export OMP_NUM_THREADS=8

tag="pretrained"

## Pretained model
if [ "$tag" == "pretrained" ]; then
  model="titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5"

## Model trained on all transitions + higher lr + higher pressure coef + simultaneous pair selection ###
elif [ "$tag" == "refined" ]; then
  model="titanium_train_solid_MACE_r_cutoff_0.5_2025_10_31_cb29134c-07c7-4420-a0b5-b374b3cd7463"
fi

# Run at experimental temperature
T=("1154.7" "1124.5" "1094.4" "1064.3" "1034.2" "1005.0")
P=(0 1 2 3 4 5)

# Run job.
for n in "${!T[@]}"
do
  folderpath="./solidfe/output/hcp-bcc/${model}/prepare_T_${T[n]}_P_${P[n]}/"
  mkdir -p $folderpath # Create directory structure for data output.

  RANDOM="13244" # Generate a random number for each run.

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

    # Create and equilibrate bcc structure.
  echo "Run HCP Preparation"
  lmp       -in lammps/prepare_hcp.lmp  \
            -log ${folderpath}/prepare_hcp.log \
            -var model "$model"          \
            -var RANDOM ${RANDOM}         \
            -var EstTemp ${T[n]}          \
            -var folderpath ${folderpath} \
            -var DefPress ${P[n]}         \
            -var structure hcp

done


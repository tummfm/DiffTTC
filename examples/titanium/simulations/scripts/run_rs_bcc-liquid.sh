# Setup list of parameters to loop over.

export OMP_NUM_THREADS=8

tag="pretrained"

## Pretrained ##############################################################
if [ "$tag" == "pretrained" ]; then
  model="titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5"
  # Run RS at experimental temperature
  T=("1940.52" "1979.41" "2011.3"  "2041.43" "2071.56" "2092.93")
  P=(0 1 2 3 4 5)

  # Run towards the coexistence temperature
  input="../../../juwels/output/coexistence/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/temps.txt"

## Refined model ##############################################################
elif [ "$tag" == "refined" ]; then
  model="titanium_train_solid_MACE_r_cutoff_0.5_2025_10_25_d1d7ab61-d1dd-4862-a944-3d7622576ab0"

  # Start at experimental temperature
  T=("1940.52" "1979.41" "2011.3"  "2041.43" "2071.56" "2092.93")
  P=(0 1 2 3 4 5)

  # TODO: Provide inputs for new model
  exit 1

fi

declare -a Ts

skip_header=true
while read -a cols; do
  if [ "$skip_header" = true ]; then
    skip_header=false
    continue
  fi
    echo "Run: ${cols[0]} GPa | Ts = ${cols[3]} K"
    Ts+=("${cols[3]}")
done < $input

# Run job.
for struct in "bcc" "liquid"; do

  for n in "${!T[@]}"; do
    folderpath="./solidfe/output/${model}/bcc-liquid/reversible_scaling_T_${T[n]}_P_${P[n]}_${struct}/"
    prepared="../../../juwels/output/bcc-liquid/${model}/prepare_T_${T[n]}_P_${P[n]}/${struct}_equilibrated_P_${P[n]}_T_${T[n]}.lmpdat"

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
    lmp       -in lammps/reversible_scaling.lmp  \
              -log ${folderpath}/reversible_scaling.log \
              -var folderpath ${folderpath} \
              -var model "$model"          \
              -var RANDOM ${RANDOM}         \
              -var T ${T[n]}          \
              -var P ${P[n]}         \
              -var Ts ${Ts[n]}       \
              -var couple "xyz"       \
              -var prepared $prepared
  done
done
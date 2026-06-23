# Setup list of parameters to loop over.

export OMP_NUM_THREADS=8

tag="pretrained"

## Pretrained ##############################################################
if [ "$tag" == "pretrained" ]; then
  model="titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5"

  input="../../../juwels/output/hcp-bcc/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/hcp_constants.txt"

## Refined model ##############################################################
elif [ "$tag" == "refined_all" ]; then
model="titanium_train_solid_MACE_r_cutoff_0.5_2025_10_25_d1d7ab61-d1dd-4862-a944-3d7622576ab0"

fi

declare -a T P a ca k

skip_header=true
while read -a cols; do
  if [ "$skip_header" = true ]; then
    skip_header=false
    continue
  fi
    echo "Run: ${cols[0]} GPa | ${cols[1]} K"
    P+=("${cols[0]}")
    T+=("${cols[1]}")
    a+=("${cols[2]}")
    ca+=("${cols[3]}")
    k+=("${cols[4]}")
done < $input


# Run job.
for n in "${!T[@]}"
do
  folderpath="./solidfe/output/hcp-bcc/${model}/frenkel_ladd_T_${T[n]}_P_${P[n]}_k_${k[n]}_a_${a[n]}_ca_${ca[n]}/"
  mkdir -p $folderpath # Create directory structure for data output.

  RANDOM="13244" # Generate a random number for each run.

  # Create and equilibrate bcc structure.
  lmp       -in lammps/frenkel_ladd_hcp.lmp  \
            -log ${folderpath}/frenkel_ladd_hcp.log \
            -var folderpath ${folderpath} \
            -var model "$model"          \
            -var RANDOM ${RANDOM}         \
            -var T ${T[n]}          \
            -var k ${k[n]}         \
            -var a ${a[n]}         \
            -var ca ${ca[n]}
done

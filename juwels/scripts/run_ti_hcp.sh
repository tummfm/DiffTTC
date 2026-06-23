# Setup list of parameters to loop over.

export OMP_NUM_THREADS=8

tag="refined"

## Pretrained ##############################################################
if [ "$tag" == "pretrained" ]; then
  model="titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5"

elif [ "$tag" == "refined" ]; then
  model="titanium_train_solid_MACE_r_cutoff_0.5_2025_11_13_f15304d6-4928-486b-b4e7-1fa652dadbff"

fi

input="${SCRATCH}/projects/crystal_fem/solidfe/hcp-bcc/${model}/hcp_constants.txt"

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
  folderpath="${SCRATCH}/projects/crystal_fem/solidfe/hcp-bcc/${model}/frenkel_ladd_T_${T[n]}_P_${P[n]}_k_${k[n]}_a_${a[n]}_ca_${ca[n]}/"
  mkdir -p $folderpath # Create directory structure for data output.

  RANDOM="13244" # Generate a random number for each run.

  # Create and equilibrate bcc structure.
  srun --nodes 1 --exclusive -n 1 --gres=gpu:1 \
  lmp       -in solidfe/frenkel_ladd_hcp.lmp  \
            -log ${folderpath}/frenkel_ladd_hcp.log \
            -var folderpath ${folderpath} \
            -var model "$model"          \
            -var RANDOM ${RANDOM}         \
            -var T ${T[n]}          \
            -var k ${k[n]}         \
            -var a ${a[n]}         \
            -var ca ${ca[n]} &
done

wait

# Setup list of parameters to loop over.

export OMP_NUM_THREADS=8
model="titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5"

# Run at experimental temperature
T=("1154.7" "1124.5" "1094.4" "1064.3" "1034.2" "1005.0")
P=(0 1 2 3 4 5)

input="../../../juwels/output/hcp-bcc/titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5/hcp_constants.txt"

declare -a a ca
skip_header=true
while read -a cols; do
  if [ "$skip_header" = true ]; then
    skip_header=false
    continue
  fi

  echo "Run: ${cols[0]} GPa | a = ${cols[2]} Å | c/a = ${cols[3]}"
  a+=("${cols[2]}")
  ca+=("${cols[3]}")
done < $input

# Run job.
for n in "${!T[@]}"
do
  folderpath="./solidfe/output/${model}/boxes/"
  mkdir -p $folderpath # Create directory structure for data output.

  # Create and equilibrate bcc structure.
  lmp       -in lammps/create_hcp_box.lmp  \
            -var folderpath ${folderpath} \
            -var model "$model"          \
            -var T ${T[n]}          \
            -var P ${P[n]}         \
            -var a ${a[n]}         \
            -var ca ${ca[n]}
done

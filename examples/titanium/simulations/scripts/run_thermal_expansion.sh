# Setup list of parameters to loop over.

export OMP_NUM_THREADS=8

tag="refined"
# tag="pretrained"

## Pretained model
if [ "$tag" == "pretrained" ]; then
  model="titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5"

elif [ "$tag" == "refined" ]; then
  model="titanium_train_solid_MACE_r_cutoff_0.5_2025_10_25_d1d7ab61-d1dd-4862-a944-3d7622576ab0"

fi

## Run at experimental temperature
P=0

# Run job.

RANDOM="13244" # Generate a random number for each run.

for struct in "hcp" "bcc" "liquid"; do
    folderpath="./solidfe/output/${model}/thermal_expansion/${struct}_thermal_expansion_P_${P}/"
    mkdir -p $folderpath # Create directory structure for data output.

    if [ "$struct" == "hcp" ]; then
      coupl="xy"
      Tstart="1200"
      Tend="0"
      tag=""

      T="1154.7"
    elif [ "$struct" == "liquid" ]; then
      Tstart="2200"
      Tend="1800"
      coupl="xyz"
      tag="melting"

      T="1940.52"
    else
      Tstart="2000"
      Tend="1000"
      coupl="xyz"
      tag="melting"

      T="1940.52"
    fi

      prepared="./solidfe/output/${model}/${tag}/prepare_T_${T[n]}_P_${P[n]}/${struct}_equilibrated_P_${P[n]}_T_${T[n]}.lmpdat"

    # Create and equilibrate hcp structure.
    lmp       -in lammps/thermal_expansion.lmp  \
              -log ${folderpath}/thermal_expansion_${struct}.log \
              -var folderpath ${folderpath} \
              -var model "$model"          \
              -var RANDOM ${RANDOM}         \
              -var T ${T}          \
              -var P ${P}         \
              -var Tstart $Tstart       \
              -var Tend $Tend         \
              -var couple $coupl       \
              -var prepared $prepared \
              -var steps "21"

done
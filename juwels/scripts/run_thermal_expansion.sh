#! /bin/bash -x
# Setup list of parameters to loop over.

#SBATCH --account=mfm
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/solidfe-bcc-hcp-gpu-out.%j
#SBATCH --error=logs/solidfe-bcc-hcp-gpu-err.%j
#SBATCH --time=05:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12

export OMP_NUM_THREADS=12

tags=("pretrained" "refined")

for tag in "${tags[@]}"; do

  ## Pretained model
  if [ "$tag" == "pretrained" ]; then
    model="titanium__MACE_r_cutoff_0.5_2025_5_4_21abc4af-aea1-4549-a9a5-981f6324c2e5"

  elif [ "$tag" == "refined" ]; then
    model="titanium_train_solid_MACE_r_cutoff_0.5_2025_11_13_f15304d6-4928-486b-b4e7-1fa652dadbff"

  fi

  P=0


  RANDOM="13244" # Generate a random number for each run.

  for struct in "hcp" "bcc" "liquid"; do
      if [ "$struct" == "hcp" ]; then
        coupl="xy"
        Tstart="1200"
        Tend="100"
        T="1154.7"

        prepared="${SCRATCH}/projects/crystal_fem/solidfe/hcp-bcc/${model}/prepare_T_${T[n]}_P_${P[n]}/${struct}_equilibrated_P_${P[n]}_T_${T[n]}.lmpdat"

      elif [ "$struct" == "liquid" ]; then
        Tstart="2200"
        Tend="1800"
        coupl="xyz"
        tag="melting"
        T="1940.52"

        prepared="${SCRATCH}/projects/crystal_fem/bcc-liquid/${model}/prepare_T_${T[n]}_P_${P[n]}/${struct}_equilibrated_P_${P[n]}_T_${T[n]}.lmpdat"

      else
        Tstart="2000"
        Tend="1000"
        coupl="xyz"
        tag="melting"
        T="1940.52"

        prepared="${SCRATCH}/projects/crystal_fem/bcc-liquid/${model}/prepare_T_${T[n]}_P_${P[n]}/${struct}_equilibrated_P_${P[n]}_T_${T[n]}.lmpdat"

      fi

      folderpath="${SCRATCH}/projects/crystal_fem/thermal_expansion/${model}/${struct}_thermal_expansion_P_${P}/"

      if [ ! -f "$prepared" ]; then
        echo "Prepared file not found: $prepared"
        exit 1
      fi

      if [ ! -d "$folderpath" ]; then
        mkdir -p $folderpath # Create directory structure for data output.
      fi

      # Create and equilibrate hcp structure.
      srun --nodes 1 --exclusive -n 2 --gres=gpu:2 \
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
                -var steps "21" \
                -var skip "12" & # Already run the first points

  done
done

wait

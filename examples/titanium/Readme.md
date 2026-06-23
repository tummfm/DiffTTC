# Titanium

## Steps

1) Pretraining models on DFT dataset
2) Matching pressure for volumes and temperatures
3) Perform initial NPT simulations to get tethering parameters
4) Perform series of 


## Training

The following scripts train and evaluate models of titanium for different
values of the cutoff.

To train run the training script and specify devices used for the run
and overwrite some hyperparameters:

```bash
python train.py 0 --cutoff 0.45
```



## Pretrained Models

Models with prior:
- titanium_MACE_r_cutoff_0.5_2025_2_15_4c6d6143-d69e-4a8d-92e5-e1df43f5eeda
- titanium_Allegro_r_cutoff_0.5_2025_2_15_36396be9-9655-48f2-84c3-d51d7da24cbb
- titanium_DimeNetPP_r_cutoff_0.5_2025_2_15_58cc7141-b352-4e9c-9cc9-ce82995be568


# Full GAN + AE Module

This folder contains full training/inference for a tabular AutoEncoder + latent conditional GAN pipeline.

## Train

```bash
python -m gan_ae_full.train --csv path/to/train.csv --out checkpoints/full_gan_ae --device cpu
```

Artifacts saved to checkpoint folder:
- `metadata.json`
- `preprocessor.pkl`
- `encoder.pt`
- `decoder.pt`
- `generator.pt`
- `discriminator.pt`
- `training_report.json`

## Use in app

In CLI/GUI, select backend `gan_ae` and set checkpoint directory to your trained folder.
The main backend will detect these artifacts and run full GAN+AE inference.


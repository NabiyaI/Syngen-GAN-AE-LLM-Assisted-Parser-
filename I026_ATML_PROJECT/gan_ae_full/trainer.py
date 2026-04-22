from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import AutoEncoder, LatentDiscriminator, LatentGenerator
from .preprocessing import TabularPreprocessor


@dataclass
class TrainConfig:
    latent_dim: int = 16
    noise_dim: int = 16
    hidden_dim: int = 128
    ae_epochs: int = 40
    gan_epochs: int = 60
    batch_size: int = 128
    lr_ae: float = 1e-3
    lr_g: float = 1e-4
    lr_d: float = 2e-4
    device: str = "cpu"
    seed: int = 42


def train_pipeline(csv_path: str, out_dir: str, config: TrainConfig) -> dict:
    _set_seed(config.seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    pre = TabularPreprocessor().fit(df)
    x = pre.transform(df)
    # Stable unconditional training: keep condition vector at zero.
    # Prompt constraints are enforced at sampling/post-filter stage.
    cond = np.zeros_like(x, dtype=np.float32)

    device = torch.device(config.device)
    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    cond_t = torch.tensor(cond, dtype=torch.float32, device=device)

    ae = AutoEncoder(input_dim=x.shape[1], latent_dim=config.latent_dim, hidden_dim=config.hidden_dim).to(device)
    ae_loss = _train_autoencoder(ae, x_t, config)

    with torch.no_grad():
        z_real = ae.encode(x_t).detach()

    generator = LatentGenerator(
        noise_dim=config.noise_dim,
        cond_dim=cond.shape[1],
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    discriminator = LatentDiscriminator(
        latent_dim=config.latent_dim,
        cond_dim=cond.shape[1],
        hidden_dim=config.hidden_dim,
    ).to(device)
    gan_report = _train_latent_gan(generator, discriminator, z_real, cond_t, config)

    torch.save(ae.encoder.state_dict(), out / "encoder.pt")
    torch.save(ae.decoder.state_dict(), out / "decoder.pt")
    torch.save(generator.state_dict(), out / "generator.pt")
    torch.save(discriminator.state_dict(), out / "discriminator.pt")
    np.save(out / "latent_bank.npy", z_real.detach().cpu().numpy().astype(np.float32))
    pre.save(str(out))

    metadata = {
        "input_dim": int(x.shape[1]),
        "cond_dim": int(cond.shape[1]),
        "latent_dim": int(config.latent_dim),
        "noise_dim": int(config.noise_dim),
        "hidden_dim": int(config.hidden_dim),
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "cond_mode": "zero",
        "latent_blend_alpha": 0.65,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = {
        "ae_final_mse": float(ae_loss),
        "gan_final_d_loss": float(gan_report["d_loss"]),
        "gan_final_g_loss": float(gan_report["g_loss"]),
        "config": asdict(config),
    }
    (out / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _train_autoencoder(ae: AutoEncoder, x: torch.Tensor, config: TrainConfig) -> float:
    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optim = torch.optim.Adam(ae.parameters(), lr=config.lr_ae)
    loss_fn = nn.MSELoss()
    ae.train()
    last_loss = 0.0
    for _ in range(config.ae_epochs):
        for (batch,) in loader:
            recon = ae(batch)
            loss = loss_fn(recon, batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
            last_loss = float(loss.item())
    return last_loss


def _train_latent_gan(
    generator: LatentGenerator,
    discriminator: LatentDiscriminator,
    z_real: torch.Tensor,
    cond: torch.Tensor,
    config: TrainConfig,
) -> dict:
    dataset = TensorDataset(z_real, cond)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    opt_g = torch.optim.Adam(generator.parameters(), lr=config.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=config.lr_d, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    device = z_real.device

    generator.train()
    discriminator.train()
    g_last = 0.0
    d_last = 0.0
    for _ in range(config.gan_epochs):
        for z_batch, c_batch in loader:
            bs = z_batch.size(0)
            real_label = torch.ones((bs, 1), device=device)
            fake_label = torch.zeros((bs, 1), device=device)

            noise = torch.randn((bs, config.noise_dim), device=device)
            z_fake = generator(noise, c_batch)

            d_real = discriminator(z_batch, c_batch)
            d_fake = discriminator(z_fake.detach(), c_batch)
            d_loss = bce(d_real, real_label) + bce(d_fake, fake_label)
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            noise2 = torch.randn((bs, config.noise_dim), device=device)
            z_fake2 = generator(noise2, c_batch)
            d_fake2 = discriminator(z_fake2, c_batch)
            g_loss = bce(d_fake2, real_label)
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            d_last = float(d_loss.item())
            g_last = float(g_loss.item())
    return {"d_loss": d_last, "g_loss": g_last}


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

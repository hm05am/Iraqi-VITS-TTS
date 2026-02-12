"""
VITS Single-GPU Training Script for Iraqi Arabic TTS.

No DDP / distributed training — runs on a single GPU (Google Colab T4).
Mixed precision via torch.amp.GradScaler for speed.

Usage:
    python train.py --config configs/iraqi_base.json --model_dir checkpoints/iraqi
"""

import os
import sys
import argparse
import json
import time
import math
import logging

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import commons
import utils
from utils import HParams
from data_utils import TextAudioLoader, TextAudioCollate
from models import SynthesizerTrn, MultiPeriodDiscriminator
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram, spec_to_mel
from text.symbols import symbols


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='VITS Training - Single GPU')
    parser.add_argument('--config', '-c', required=True, help='Path to config JSON')
    parser.add_argument('--model_dir', '-m', required=True, help='Directory to save checkpoints')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    hps = HParams(**config)
    
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    # Save config copy
    config_save_path = os.path.join(model_dir, 'config.json')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Set seed
    torch.manual_seed(hps.train.seed)
    torch.cuda.manual_seed(hps.train.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ============================================================
    # Data
    # ============================================================
    train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )

    eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # ============================================================
    # Models
    # ============================================================
    n_vocab = len(symbols)
    spec_channels = hps.data.filter_length // 2 + 1
    segment_size = hps.train.segment_size // hps.data.hop_length

    net_g = SynthesizerTrn(
        n_vocab,
        spec_channels,
        segment_size,
        **vars(hps.model)
    ).to(device)

    net_d = MultiPeriodDiscriminator(
        use_spectral_norm=hps.model.use_spectral_norm
    ).to(device)

    # Count parameters
    g_params = sum(p.numel() for p in net_g.parameters())
    d_params = sum(p.numel() for p in net_d.parameters())
    logger.info(f"Generator parameters: {g_params:,}")
    logger.info(f"Discriminator parameters: {d_params:,}")
    logger.info(f"Vocabulary size: {n_vocab}")

    # ============================================================
    # Optimizers
    # ============================================================
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    # ============================================================
    # Resume from checkpoint
    # ============================================================
    global_step = 0
    epoch_start = 1

    latest_g = utils.latest_checkpoint_path(model_dir, "G_*.pth")
    latest_d = utils.latest_checkpoint_path(model_dir, "D_*.pth")

    if latest_g is not None and latest_d is not None:
        try:
            net_g, optim_g, _, global_step = utils.load_checkpoint(latest_g, net_g, optim_g)
            net_d, optim_d, _, _ = utils.load_checkpoint(latest_d, net_d, optim_d)
            epoch_start = max(1, global_step // len(train_loader) + 1)
            logger.info(f"Resuming from step {global_step}, epoch ~{epoch_start}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
            global_step = 0
            epoch_start = 1
    else:
        logger.info("No checkpoint found. Starting from scratch.")

    # ============================================================
    # Schedulers
    # ============================================================
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay)
    # Fast-forward scheduler to current epoch
    for _ in range(epoch_start - 1):
        scheduler_g.step()
        scheduler_d.step()

    # ============================================================
    # Mixed precision
    # ============================================================
    scaler = torch.amp.GradScaler('cuda', enabled=hps.train.fp16_run)

    # ============================================================
    # TensorBoard
    # ============================================================
    writer = SummaryWriter(log_dir=os.path.join(model_dir, 'logs'))

    # ============================================================
    # Training Loop
    # ============================================================
    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info(f"  Batch size: {hps.train.batch_size}")
    logger.info(f"  FP16: {hps.train.fp16_run}")
    logger.info(f"  Training samples: {len(train_dataset)}")
    logger.info(f"  Validation samples: {len(eval_dataset)}")
    logger.info(f"  Steps per epoch: {len(train_loader)}")
    logger.info("=" * 60)

    net_g.train()
    net_d.train()

    for epoch in range(epoch_start, hps.train.epochs + 1):
        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Unpack batch
            x, x_lengths, spec, spec_lengths, y = batch
            x = x.to(device, non_blocking=True)
            x_lengths = x_lengths.to(device, non_blocking=True)
            spec = spec.to(device, non_blocking=True)
            spec_lengths = spec_lengths.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # ========================================
            # Generator forward
            # ========================================
            with torch.amp.autocast('cuda', enabled=hps.train.fp16_run):
                (y_hat, l_length, attn, ids_slice, x_mask, z_mask,
                 (z, z_p, m_p, logs_p, m_q, logs_q)) = net_g(x, x_lengths, spec, spec_lengths)

                # Mel loss
                mel = spec_to_mel(
                    spec, hps.data.filter_length, hps.data.n_mel_channels,
                    hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax
                )
                y_mel = commons.slice_segments(
                    mel, ids_slice, segment_size
                )

                y_hat_mel = mel_spectrogram(
                    y_hat.squeeze(1), hps.data.filter_length,
                    hps.data.n_mel_channels, hps.data.sampling_rate,
                    hps.data.hop_length, hps.data.win_length,
                    hps.data.mel_fmin, hps.data.mel_fmax
                )

                # Slice waveform for discriminator
                y_slice = commons.slice_segments(
                    y, ids_slice * hps.data.hop_length,
                    hps.train.segment_size
                )

                # Discriminator forward
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y_slice, y_hat.detach())
                with torch.amp.autocast('cuda', enabled=False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        y_d_hat_r, y_d_hat_g
                    )
                    loss_disc_all = loss_disc

            # ========================================
            # Discriminator backward
            # ========================================
            optim_d.zero_grad()
            scaler.scale(loss_disc_all).backward()
            scaler.unscale_(optim_d)
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)

            # ========================================
            # Generator losses
            # ========================================
            with torch.amp.autocast('cuda', enabled=hps.train.fp16_run):
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y_slice, y_hat)
                with torch.amp.autocast('cuda', enabled=False):
                    loss_dur = torch.sum(l_length.float())
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

            # ========================================
            # Generator backward
            # ========================================
            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()

            global_step += 1

            # ========================================
            # Logging
            # ========================================
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                logger.info(
                    f"Step {global_step} | Epoch {epoch} | "
                    f"loss_g: {loss_gen_all.item():.3f} | "
                    f"loss_d: {loss_disc_all.item():.3f} | "
                    f"loss_mel: {loss_mel.item():.3f} | "
                    f"loss_dur: {loss_dur.item():.3f} | "
                    f"loss_kl: {loss_kl.item():.3f} | "
                    f"grad_g: {grad_norm_g:.3f} | "
                    f"grad_d: {grad_norm_d:.3f} | "
                    f"lr: {lr:.6f}"
                )

                # === ALIGNMENT CHECK ===
                if loss_dur.item() < 1.0:
                    logger.info("  ✅ Alignment appears to be converging (loss_dur < 1.0)")
                elif loss_dur.item() < 5.0:
                    logger.info("  ⏳ Alignment is progressing (loss_dur < 5.0)")
                else:
                    logger.info("  ⚠️  Alignment not yet converged (loss_dur >= 5.0)")

                # TensorBoard
                utils.summarize(writer, global_step, scalars={
                    'loss/g/total': loss_gen_all.item(),
                    'loss/d/total': loss_disc_all.item(),
                    'loss/g/mel': loss_mel.item(),
                    'loss/g/dur': loss_dur.item(),
                    'loss/g/kl': loss_kl.item(),
                    'loss/g/fm': loss_fm.item(),
                    'learning_rate': lr,
                    'grad_norm/g': grad_norm_g,
                    'grad_norm/d': grad_norm_d,
                })

            # ========================================
            # Checkpoint saving
            # ========================================
            if global_step % hps.train.eval_interval == 0:
                logger.info(f"Saving checkpoint at step {global_step}...")
                utils.save_checkpoint(
                    net_g, optim_g, hps.train.learning_rate, global_step,
                    os.path.join(model_dir, f"G_{global_step}.pth")
                )
                utils.save_checkpoint(
                    net_d, optim_d, hps.train.learning_rate, global_step,
                    os.path.join(model_dir, f"D_{global_step}.pth")
                )

                # Run evaluation
                evaluate(net_g, eval_loader, hps, writer, global_step, device)

        # End of epoch
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch} complete in {epoch_time:.1f}s")
        scheduler_g.step()
        scheduler_d.step()

    writer.close()
    logger.info("Training complete!")


def evaluate(net_g, eval_loader, hps, writer, global_step, device):
    """Run evaluation and log sample spectrograms."""
    net_g.eval()
    segment_size = hps.train.segment_size // hps.data.hop_length
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            x, x_lengths, spec, spec_lengths, y = batch
            x = x.to(device)
            x_lengths = x_lengths.to(device)
            spec = spec.to(device)
            spec_lengths = spec_lengths.to(device)
            y = y.to(device)

            # Run inference
            y_hat, attn, mask, _ = net_g.infer(x, x_lengths, max_len=1000)
            y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

            # Log first sample's spectrogram
            if batch_idx == 0:
                y_hat_mel = mel_spectrogram(
                    y_hat.squeeze(1).float(), hps.data.filter_length,
                    hps.data.n_mel_channels, hps.data.sampling_rate,
                    hps.data.hop_length, hps.data.win_length,
                    hps.data.mel_fmin, hps.data.mel_fmax
                )
                image = utils.plot_spectrogram_to_numpy(
                    y_hat_mel[0].cpu().numpy()
                )
                utils.summarize(writer, global_step, images={
                    'eval/mel_predicted': image,
                })

                # Log attention
                if attn is not None:
                    attn_img = utils.plot_spectrogram_to_numpy(
                        attn[0, 0].cpu().numpy()
                    )
                    utils.summarize(writer, global_step, images={
                        'eval/attention': attn_img,
                    })
            
            if batch_idx >= 3:
                break

    net_g.train()


if __name__ == '__main__':
    main()

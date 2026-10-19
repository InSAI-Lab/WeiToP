"""Joint training of WeiAD and WeiToP on GSV-Cities."""

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_metric_learning import distances, losses, miners

if __package__:
    from .dataloader import GSVCitiesDataModule
    from .model import WeiToP
else:
    from dataloader import GSVCitiesDataModule
    from model import WeiToP


class TrainingModule(pl.LightningModule):
    """Optimise global descriptors and early token-importance predictions."""

    def __init__(
        self,
        dinov2_path: str | None = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = WeiToP(dinov2_path=dinov2_path, pretrained=pretrained)
        # Learn global descriptors from mined positive and negative pairs.
        self.retrieval_loss = losses.MultiSimilarityLoss(
            alpha=1.0, beta=20.0, base=0.0,
            distance=distances.DotProductSimilarity(),
        )
        self.miner = miners.MultiSimilarityMiner(
            epsilon=0.1, distance=distances.CosineSimilarity()
        )

    def forward(self, images: torch.Tensor):
        return self.model(images)

    def training_step(self, batch, batch_index):
        images, labels = batch
        # The loader batches places; retrieval learning treats all four views
        # as separate samples sharing a label, with other places as negatives.
        images = images.flatten(0, 1)
        labels = labels.flatten()
        descriptors, distillation_loss = self(images)
        with torch.autocast(device_type=images.device.type, enabled=False):
            pairs = self.miner(descriptors.float(), labels)
            retrieval_loss = self.retrieval_loss(descriptors.float(), labels, pairs)
            loss = retrieval_loss + 0.1 * distillation_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss.")
        self.log_dict(
            {"train/loss": loss, "train/retrieval": retrieval_loss,
             "train/distillation": distillation_loss},
            on_step=True, on_epoch=True, prog_bar=True, batch_size=images.shape[0],
        )
        return loss

    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=6e-5,
            weight_decay=9.5e-9,
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimiser, start_factor=1.0, end_factor=0.2, total_iters=4000
        )
        # Advance the learning-rate schedule once per optimiser update.
        return {
            "optimizer": optimiser,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train WeiAD and WeiToP on GSV-Cities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, required=True, help="GSV-Cities root.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/weitop"))
    parser.add_argument("--batch-size", type=int, default=60, help="Places per batch; four images per place.")
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--device", default="cuda:0", help="cuda:<index> or cpu.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dinov2-path", type=Path, help="Optional local DINOv2 source checkout.")
    parser.add_argument("--resume", type=Path, help="Resume training from a checkpoint.")
    args = parser.parse_args(argv)
    if args.batch_size < 2 or args.num_workers < 0 or args.epochs < 1:
        parser.error("Require batch-size >= 2, num-workers >= 0 and epochs >= 1.")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use --device cpu or --device cuda:<index>.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a smoke check.")
    pl.seed_everything(args.seed, workers=True)
    datamodule = GSVCitiesDataModule(
        root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers
    )
    module = TrainingModule(
        dinov2_path=str(args.dinov2_path) if args.dinov2_path else None,
        # Resuming restores the full backbone; downloading pretrained weights
        # again is unnecessary. Initial training always uses pretrained DINOv2.
        pretrained=args.resume is None,
    )
    checkpoint = ModelCheckpoint(
        dirpath=args.output_dir / "checkpoints",
        filename="epoch-{epoch:02d}", auto_insert_metric_name=False,
        save_last=True, save_top_k=-1, every_n_epochs=1,
    )
    trainer = pl.Trainer(
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=[device.index if device.index is not None else 0] if device.type == "cuda" else 1,
        precision="16-mixed" if device.type == "cuda" else "32-true",
        max_epochs=args.epochs,
        default_root_dir=args.output_dir,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step")],
        logger=CSVLogger(str(args.output_dir), name="logs"),
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=20,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=args.resume)


if __name__ == "__main__":
    main()

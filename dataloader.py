"""GSV-Cities place sampling and image preprocessing."""

from pathlib import Path
from collections.abc import Sequence

import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


TRAIN_CITIES = (
    "Bangkok", "BuenosAires", "LosAngeles", "MexicoCity", "OSL", "Rome",
    "Barcelona", "Chicago", "Madrid", "Miami", "Phoenix", "TRT", "Boston",
    "Lisbon", "Medellin", "Minneapolis", "PRG", "WashingtonDC", "Brussels",
    "London", "Melbourne", "Osaka", "PRS",
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGES_PER_PLACE = 4


def image_transform(training: bool = False) -> T.Compose:
    """224 px and RandAugment for training; 322 px for descriptor extraction."""
    size = (224, 224) if training else (322, 322)
    operations = [T.Resize(size, interpolation=T.InterpolationMode.BILINEAR)]
    if training:
        operations.append(T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR))
    operations.extend([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose(operations)


class GSVCitiesDataset(Dataset):
    """Return four random images of one place and their shared class label.

    The root must contain Dataframes/<city>.csv and Images/<city>/*.jpg.
    Each item is (images [4, 3, 224, 224], labels [4]). Place IDs are scoped
    by city, so equal numeric IDs in different cities never share a label.
    """

    def __init__(
        self,
        root: str | Path,
        cities: Sequence[str] = TRAIN_CITIES,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.cities = tuple(cities)
        if not self.cities or len(set(self.cities)) != len(self.cities):
            raise ValueError("cities must contain distinct city names.")
        self.transform = image_transform(training=True)
        self._frames: list[pd.DataFrame] = []
        self._places: list[tuple[int, object]] = []
        self.num_images = 0
        required = {
            "place_id", "year", "month", "northdeg", "city_id", "lat", "lon", "panoid"
        }
        for city_index, city in enumerate(self.cities):
            csv_path = self.root / "Dataframes" / f"{city}.csv"
            if not csv_path.is_file():
                raise FileNotFoundError(f"Missing GSV-Cities metadata: {csv_path}")
            # Preserve coordinates as text: converting them to floats can
            # change the filenames when their decimal representation rounds.
            frame = pd.read_csv(
                csv_path, dtype={"city_id": str, "panoid": str, "lat": str, "lon": str}
            )
            if missing := required.difference(frame.columns):
                raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
            if frame[list(required)].isna().any().any():
                raise ValueError(f"{csv_path} contains missing image metadata.")
            if not frame["city_id"].eq(city).all():
                raise ValueError(f"city_id values in {csv_path} must match {city}.")
            if not (self.root / "Images" / city).is_dir():
                raise FileNotFoundError(f"Missing GSV-Cities image directory: {self.root / 'Images' / city}")
            frame = frame[
                frame.groupby("place_id")["place_id"].transform("size") >= IMAGES_PER_PLACE
            ]
            # Shuffle places within each city, retaining the city order.
            # The DataModule rebuilds these groups at the start of each epoch.
            frame = frame.sample(frac=1).reset_index(drop=True)
            self._frames.append(frame)
            self._places.extend(
                (city_index, rows)
                for rows in frame.groupby("place_id", sort=False).indices.values()
            )
            self.num_images += len(frame)
        if not self._places:
            raise ValueError("No places contain at least four images.")

    def __len__(self) -> int:
        return len(self._places)

    @staticmethod
    def image_name(row: pd.Series) -> str:
        """Reconstruct the image filename using the city-local place ID."""
        return (
            f"{row['city_id']}_{int(row['place_id']):07d}_"
            f"{int(row['year']):04d}_{int(row['month']):02d}_"
            f"{int(row['northdeg']):03d}_{row['lat']}_{row['lon']}_{row['panoid']}.jpg"
        )

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        city_index, rows = self._places[index]
        chosen = torch.randperm(len(rows))[:IMAGES_PER_PLACE].tolist()
        place = self._frames[city_index].iloc[rows[chosen]]
        images = []
        for _, row in place.iterrows():
            path = self.root / "Images" / row["city_id"] / self.image_name(row)
            # Include the image path when reporting missing or corrupt data.
            try:
                with Image.open(path) as image:
                    images.append(self.transform(image.convert("RGB")))
            except (OSError, ValueError) as error:
                raise RuntimeError(f"Could not load GSV-Cities image: {path}") from error
        labels = torch.full((IMAGES_PER_PLACE,), index, dtype=torch.long)
        return torch.stack(images), labels


class GSVCitiesDataModule(pl.LightningDataModule):
    """Each batch contains 60 places with four views per place by default."""

    def __init__(
        self,
        root: str | Path,
        batch_size: int = 60,
        num_workers: int = 10,
        cities: Sequence[str] = TRAIN_CITIES,
    ) -> None:
        super().__init__()
        if batch_size < 2 or num_workers < 0:
            raise ValueError("batch_size must be at least 2; num_workers must be non-negative.")
        self.save_hyperparameters()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cities = tuple(cities)

    def train_dataloader(self) -> DataLoader:
        dataset = GSVCitiesDataset(self.root, self.cities)
        if len(dataset) < self.batch_size:
            raise ValueError("The dataset has fewer places than batch_size; reduce batch_size.")
        # A singleton final batch lacks a negative place for retrieval learning.
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=len(dataset) % self.batch_size == 1,
            num_workers=self.num_workers,
            pin_memory=True,
        )

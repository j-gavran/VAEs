import numpy as np
import pytorch_lightning as pl
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from vaes.data_utils.dataset_utils import Dataset, rescale_data


class DummyDataModule(pl.LightningDataModule):
    def __init__(self, data_size=None, batch_size=None, to_gpu=False, **dl_kwargs):
        """Dummy pl data module for testing.

        Parameters
        ----------
        data_size : tuple
            Size of data matrix X. First feature column is reserved for label.
        batch_size : int
            Batch size.
        to_gpu : bool, optional
            If True put data on gpu, by default False.
        dl_kwargs: **kwargs
            Additional parameters for torch DataLoader class. See https://pytorch.org/docs/stable/data.html .

        References
        ----------
        - https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html

        """
        super().__init__()
        self.data_size = data_size
        self.batch_size = batch_size
        self.to_gpu = to_gpu
        self.dl_kwargs = dl_kwargs
        self.data, self.labels = None, None

    def prepare_data(self):
        """Creates a list [train, val, test] of numpy float32 normaly distributed X matrices."""
        self.data = [np.random.normal(loc=0, scale=1, size=self.data_size).astype(np.float32) for _ in range(3)]
        self.labels = [np.zeros(self.data_size) for _ in range(3)]

    def setup(self, stage=None):
        """Creates train, val and test datasets."""
        if stage == "fit" or stage is None:
            self.train = Dataset(self.data[0], self.labels[0], to_gpu=self.to_gpu)
            self.val = Dataset(self.data[1], self.labels[1], to_gpu=self.to_gpu)
            if len(self.data) > 2:
                self.test = Dataset(self.data[2], self.labels[2], to_gpu=self.to_gpu)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, drop_last=True, shuffle=True, **self.dl_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, drop_last=True, shuffle=False, **self.dl_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, drop_last=True, shuffle=False, **self.dl_kwargs)


class CustomDummyDataModule(DummyDataModule):
    def __init__(
        self,
        data_dir=None,
        data_size=None,
        batch_size=128,
        to_gpu=False,
        rescale=False,
        subset_n=None,
        shuffle_data=False,
        train_val_split=None,
        dequantize=False,
        **dl_kwargs,
    ):
        """Custom template dataset DataModule (serves as a base for all other data modules).

        Warning
        -------
        0th column is reserved for labels!

        Parameters
        ----------
        data_dir : str or list of str
            List of train/val/test data locations or string to file location.
        batch_size : int, optional
            Batch size, by default 128.
        rescale : str, optional
           See data_utils.dataset_utils, by default None.
        subset_n : int, optional
            Subset of data to consider, by default None.
        to_gpu : bool, optional
            Put data directly to gpu memory, by default False.
        shuffle_data : bool, optional
            Shuffle arrays of data, by default False.
        train_val_split : float in (0, 1), optional
            If not None splits data into partitions defined by train_val_split %. Can only be used if only one data_dir file given.
        dequantize : bool, optional.
            Add random unifrom noise to data before rescaling.
        """
        super().__init__(data_size, batch_size, to_gpu, **dl_kwargs)
        self.data_dir = data_dir
        self.rescale = rescale
        self.subset_n = subset_n
        self.shuffle = shuffle_data
        self.train_val_split = train_val_split
        self.dequantize = dequantize

        self.load_data()

        self.scalers = []  # save list for sklearn scalers

    def load_data(self):
        """Load npy files into lists of train/val/test arrays."""
        if type(self.data_dir) is str:
            self.data_dir = [self.data_dir]

        if self.train_val_split is not None:
            assert len(self.data_dir) == 1
            assert 0.0 < self.train_val_split < 1.0
            data = np.load(self.data_dir[0]).astype(np.float32)
            idx = int(len(data) * self.train_val_split)
            self.data = [data[:idx, :], data[idx:, :]]
        else:
            self.data = [np.load(d).astype(np.float32) for d in self.data_dir]

        self.labels = [self.data[i][:, 0] for i in range(len(self.data))]

        self.subset_n = self.subset_n if self.subset_n is not None else [None] * len(self.data)

    def prepare_data(self):
        for i, s in enumerate(self.subset_n):
            if s is not None:
                assert s <= len(self.data[i])
                self.data[i] = self.data[i][:s]

        if self.shuffle:
            for i in range(len(self.data)):
                self.data[i], self.labels[i] = shuffle(self.data[i], self.labels[i])

        if self.dequantize:
            for i in range(len(self.data)):
                norm = np.max(np.abs(self.data[i])).astype(np.float32)
                self.data[i] = self.data[i] + np.random.uniform(size=self.data[i].shape).astype(np.float32) / norm
                self.data[i] = self.data[i].clip(0, 1)

        if self.rescale:
            # 0th column are labels that we dont want to normalize!
            for i in range(len(self.data)):
                self.data[i][:, 1:], scaler = rescale_data(self.data[i][:, 1:], rescale_type=self.rescale)
                self.scalers.append(scaler)

    def _get_data_shape(self):
        data = [np.load(d) for d in self.data_dir]
        return [i.shape for i in data]

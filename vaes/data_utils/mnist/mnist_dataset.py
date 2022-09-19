import numpy as np
from vaes.data_utils.dummy_dataset import CustomDummyDataModule


class MnistDataModule(CustomDummyDataModule):
    def __init__(
        self,
        data_dir=None,
        data_size=None,
        batch_size=128,
        to_gpu=False,
        rescale=False,
        subset_n=None,
        shuffle_data=False,
        dequantize=True,
        flatten=True,
        data_labels=None,
        **dl_kwargs
    ):
        """MNIST DataModule.

        Parameters
        ----------
        flatten : bool, optional.
            Flattens image to 1d array, by default True.
        data_labels : list of str, optional.
            List of paths to labels (if None all zero), by default None.
        """
        super().__init__(
            data_dir=data_dir,
            data_size=data_size,
            batch_size=batch_size,
            to_gpu=to_gpu,
            rescale=rescale,
            subset_n=subset_n,
            shuffle_data=shuffle_data,
            dequantize=dequantize,
            **dl_kwargs
        )

        # reshape
        if flatten:
            for i in range(len(self.data)):
                self.data[i] = self.data[i].reshape(
                    self.data[i].shape[0], self.data[i].shape[1] * self.data[i].shape[2]
                )

        # add labels
        self.data_labels = data_labels
        if data_labels:
            self.labels = [np.load(l).astype(np.int64) for l in data_labels]


if __name__ == "__main__":
    mnist_data = MnistDataModule(
        [
            "data/mnist/train.npy",
            "data/mnist/test.npy",
        ],
        batch_size=1024,
        num_workers=12,
        rescale=None,
        dequantize=True,
        shuffle_data=True,
        data_labels=[
            "data/mnist/train_labels.npy",
            "data/mnist/test_labels.npy",
        ],
    )

    idx = -1

    print(mnist_data._get_data_shape())

    mnist_data.prepare_data()
    mnist_data.setup()
    s = mnist_data.train.X[idx]

    print(s.dtype)

    print(mnist_data.train.y[idx])
    # print(mnist_data.train.X[idx])

    import matplotlib.pyplot as plt

    plt.imshow(s.reshape(28, 28))
    plt.colorbar()
    plt.show()

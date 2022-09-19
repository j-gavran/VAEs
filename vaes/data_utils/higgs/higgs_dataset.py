import numpy as np
from vaes.data_utils.dummy_dataset import CustomDummyDataModule


class HiggsDataModule(CustomDummyDataModule):
    def __init__(
        self,
        data_dir=None,
        batch_size=128,
        to_gpu=False,
        rescale=False,
        subset_n=None,
        shuffle_data=False,
        **dl_kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            to_gpu=to_gpu,
            rescale=rescale,
            subset_n=subset_n,
            shuffle_data=shuffle_data,
            **dl_kwargs,
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    _data = [
        "data/higgs/HIGGS_18_feature_bkg_train.npy",
        "data/higgs/HIGGS_18_feature_bkg_val.npy",
        "data/higgs/HIGGS_18_feature_bkg_test.npy",
    ]

    higgs_data = HiggsDataModule(
        _data,
        batch_size=1024,
        rescale=None,
        num_workers=12,
        subset_n=[10 ** 5, 10 ** 5, 10 ** 5],
        shuffle_data=True,
    )

    higgs_data.prepare_data()
    higgs_data.setup()
    train_data = higgs_data.train.X.numpy()

    fig, axs = plt.subplots(6, 3)
    axs = axs.flatten()

    for i in range(len(axs)):
        axs[i].hist(train_data[:, i], bins=40, histtype="step")

    plt.show()

    print(np.unique(higgs_data.train.y, return_counts=True))

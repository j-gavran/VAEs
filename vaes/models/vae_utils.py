import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from vaes.data_utils.toy_datasets import TOY_DATASETS


def two_sample_plot(
    A,
    B,
    axs,
    n_bins="auto",
    label=None,
    labels=None,
    log_scale=False,
    bin_range=None,
    xlim=None,
    ylim=None,
    titles=None,
    **kwargs,
):
    assert A.shape[1] == B.shape[1]

    n_features = A.shape[1]

    if bin_range is not None:
        if not any(isinstance(el, list) for el in bin_range):
            bin_range = [bin_range] * n_features

    if torch.is_tensor(A) and torch.is_tensor(B):
        A, B = A.numpy(), B.numpy()

    combined_sample = np.concatenate([A, B])

    for feature in range(n_features):
        bin_edges = np.histogram_bin_edges(
            combined_sample[:, feature], bins=n_bins, range=bin_range[feature] if bin_range else None
        )

        axs[feature].hist(A[:, feature], bins=bin_edges, histtype="step", **kwargs)
        axs[feature].hist(B[:, feature], bins=bin_edges, histtype="step", **kwargs)

        if feature == 0 and label is not None:
            axs[feature].legend(label)

        if labels is not None:
            axs[feature].set_xlabel(labels[feature], size=15)

        if log_scale:
            axs[feature].set_yscale("log")

        if xlim:
            axs[feature].set_xlim(xlim[feature])

        if ylim:
            if ylim[feature] is not None:
                axs[feature].set_ylim(ylim[feature])

        if titles is not None:
            axs[feature].set_title(titles[feature], size=15, loc="right")

    return axs


def plot_vae_result_on_event(data_name, model, logger, test_dataloader, idx=0, use_hexbin=False):

    if 0 < idx < 100:
        idx = f"0{idx}"

    def _vae_on_train_end():
        if data_name.lower() == "mnist":
            n = 10
            fig, axs = plt.subplots(n, n, figsize=(15, 15))
            axs = axs.flatten()

            x = model.sample(n * n, device=model.device).cpu().numpy()

            for i in range(n * n):
                axs[i].imshow(x[i, :].reshape(28, 28).clip(0, 1), cmap="gray")

            plt.tight_layout()
            logger.experiment.log_figure(logger.run_id, fig, f"MNIST_generated_{idx}.jpg")

        elif data_name.lower() in ["higgs", "higgs_bkg", "higgs_sig"]:
            subset = 10 ** 5
            X = test_dataloader().dataset.X[:subset].cpu().numpy()
            sample = model.sample(subset, device=model.device).cpu().numpy()

            scalers = model.scalers
            X = scalers[2].inverse_transform(X)
            sample = scalers[0].inverse_transform(sample)

            fig, axs = plt.subplots(6, 3, figsize=(10, 10))
            axs = axs.flatten()

            sample = sample[~np.isnan(sample).any(axis=1)]

            two_sample_plot(X, sample, axs, n_bins=50, log_scale=False, density=True, lw=2)

            plt.tight_layout()
            logger.experiment.log_figure(logger.run_id, fig, f"Higgs_generated_{idx}.jpg")

            fig, axs = plt.subplots(6, 3, figsize=(10, 10))
            axs = axs.flatten()

            two_sample_plot(X, sample, axs, n_bins=50, log_scale=True, density=False, lw=2)

            plt.tight_layout()
            logger.experiment.log_figure(logger.run_id, fig, f"Higgs_generated_log_{idx}.jpg")

        elif data_name in TOY_DATASETS:
            subset = 3 * 10 ** 4
            generated = model.sample(subset, device=model.device).cpu().numpy()

            fig, axs = plt.subplots(1, 1, figsize=(10, 8))
            if use_hexbin:
                axs.hexbin(generated[:, 0], generated[:, 1], cmap="jet", extent=[-4, 4, -4, 4], gridsize=150)
            else:
                axs.scatter(generated[:, 0], generated[:, 1], s=0.25)

            axs.set_title("generated")

            logger.experiment.log_figure(logger.run_id, fig, f"2d_test_{idx}.jpg")
        else:
            logging.warning("data_name not implemented...")

    try:
        with torch.no_grad():
            model.eval()
            _vae_on_train_end()
            model.train()

            plt.tight_layout()
            plt.close()
    except Exception as e:
        logging.warning(f"plotting quit with exception {e}")

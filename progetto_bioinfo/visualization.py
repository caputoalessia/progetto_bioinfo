from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from multiprocessing import cpu_count
from cache_decorator import Cache
from glob import glob
import seaborn as sns
from sklearn.decomposition import PCA
from prince import MFA
from MulticoreTSNE import MulticoreTSNE as UTSNE
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from minepy import MINE
from scipy.stats import spearmanr
from keras_bed_sequence import BedSequence
from ucsc_genomes_downloader import Genome


def to_bed(data: pd.DataFrame) -> pd.DataFrame:
    """Return bed coordinates from given dataset."""
    return data.reset_index()[data.index.names]


def one_hot_encode(genome: Genome, data: pd.DataFrame, nucleotides: str = "actg") -> np.ndarray:
    return np.array(BedSequence(
        genome,
        bed=to_bed(data),
        nucleotides=nucleotides,
        batch_size=1
    ))


def flat_one_hot_encode(genome: Genome, data: pd.DataFrame, window_size: int, nucleotides: str = "actg") -> np.ndarray:
    return one_hot_encode(genome, data, nucleotides).reshape(-1, window_size*4).astype(int)


def to_dataframe(x: np.ndarray, window_size: int, nucleotides: str = "actg") -> pd.DataFrame:
    return pd.DataFrame(
        x,
        columns=[
            f"{i}{nucleotide}"
            for i in range(window_size)
            for nucleotide in nucleotides
        ]
    )

# Decomposizopme PCA
def pca(x: np.ndarray, n_components: int = 2) -> np.ndarray:
    return PCA(n_components=n_components, random_state=42).fit_transform(x)

#TSNE con decomposizione se necessario
def ulyanov_tsne(x: np.ndarray, perplexity: int, dimensionality_threshold: int = 50, n_components: int = 2):
    if x.shape[1] > dimensionality_threshold:
        x = pca(x, n_components=dimensionality_threshold)
    return UTSNE(n_components=n_components, perplexity=perplexity, n_jobs=cpu_count(), random_state=42, verbose=True).fit_transform(x)


def visualize(cell_line, epigenomes, labels):
    genome = Genome("hg19")
    sequences = {
        region: to_dataframe(
            flat_one_hot_encode(genome, data, 200),
            200
        )
        for region, data in epigenomes.items()
    }
    tasks = {
        "x": [
            *[
                val.values
                for val in epigenomes.values()
            ],
            *[
                val.values
                for val in sequences.values()
            ]
        ],
        "y": [
            *[
                val.values.ravel()
                for val in labels.values()
            ],
            *[
                val.values.ravel()
                for val in labels.values()
            ]
        ],
        "titles": [
            "Epigenomes promoters",
            "Epigenomes enhancers",
            "Sequences promoters",
            "Sequences enhancers"
        ]
    }

    xs = tasks["x"]
    ys = tasks["y"]
    titles = tasks["titles"]

    assert len(xs) == len(ys) == len(titles)

    for x, y in zip(xs, ys):
        assert x.shape[0] == y.shape[0]
    print("test")
    colors = np.array([
        "tab:blue",
        "tab:orange",
    ])

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(32, 8))

    for x, y, title, axis in tqdm(zip(xs, ys, titles, axes.flatten()), desc="Computing PCAs", total=len(xs)):
        axis.scatter(*pca(x).T, s=1, color=colors[y])
        axis.xaxis.set_visible(False)
        axis.yaxis.set_visible(False)
        axis.set_title(f"PCA decomposition - {title}")
    plt.savefig("./imgs/" + cell_line + "/PCA decomposition")
    plt.show()

    for perpexity in tqdm((50, 500), desc="Running perplexities"):
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(40, 10))
        for x, y, title, axis in tqdm(zip(xs, ys, titles, axes.flatten()), desc="Computing TSNEs", total=len(xs)):
            axis.scatter(*ulyanov_tsne(x, perplexity=perpexity).T,
                         s=1, color=colors[y])
            axis.xaxis.set_visible(False)
            axis.yaxis.set_visible(False)
            axis.set_title(f"TSNE decomposition - {title}")
        fig.tight_layout()
        fig.savefig("./imgs/" + cell_line + f"/TSNE_" + str(perpexity))
        plt.show()

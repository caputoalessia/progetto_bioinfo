from tqdm.auto import tqdm # A simple loading bar
import matplotlib.pyplot as plt # A standard plotting library
import pandas as pd
import numpy as np
from multiprocessing import cpu_count
from cache_decorator import Cache
from glob import glob 
import seaborn as sns
from epigenomic_dataset import load_epigenomes
from ucsc_genomes_downloader import Genome
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from keras_bed_sequence import BedSequence

def knn_imputer(df:pd.DataFrame, neighbours:int=5)->pd.DataFrame:
    return pd.DataFrame(
        KNNImputer(n_neighbors=neighbours).fit_transform(df.values),
        columns=df.columns,
        index=df.index
    )

def one_hot_encode(genome:Genome, data:pd.DataFrame, nucleotides:str="actg")->np.ndarray:
    return np.array(BedSequence(
        genome,
        bed=to_bed(data),
        nucleotides=nucleotides,
        batch_size=1
    ))

def flat_one_hot_encode(genome:Genome, data:pd.DataFrame, window_size:int, nucleotides:str="actg")->np.ndarray:
    return one_hot_encode(genome, data, nucleotides).reshape(-1, window_size*4).astype(int)

def to_dataframe(x:np.ndarray, window_size:int, nucleotides:str="actg")->pd.DataFrame:
    return pd.DataFrame(
        x,
        columns = [
            f"{i}{nucleotide}"
            for i in range(window_size)
            for nucleotide in nucleotides
        ]
    )

def to_bed(data:pd.DataFrame)->pd.DataFrame:
    """Return bed coordinates from given dataset."""
    return data.reset_index()[data.index.names]

def retrive_cell_line(line, win_size):
    cell_line = line
    window_size = win_size

    promoters_epigenomes, promoters_labels = load_epigenomes(
        cell_line = cell_line,
        dataset = "fantom",
        regions = "promoters",
        window_size = window_size
    )

    enhancers_epigenomes, enhancers_labels = load_epigenomes(
        cell_line = cell_line,
        dataset = "fantom",
        regions = "enhancers",
        window_size = window_size
    )

    promoters_epigenomes = promoters_epigenomes.droplevel(1, axis=1) 
    enhancers_epigenomes = enhancers_epigenomes.droplevel(1, axis=1)
    promoters_labels = promoters_labels.values.ravel()
    enhancers_labels = enhancers_labels.values.ravel()
    epigenomes = {
        "promoters": promoters_epigenomes,
        "enhancers": enhancers_epigenomes
    }
    labels = {
        "promoters": promoters_labels,
        "enhancers": enhancers_labels
    }

    #Ratio should be > 1
    for region, x in epigenomes.items():
        print(
            f"The rate between features and samples for {region} data is: {x.shape[0]/x.shape[1]}"
        )
        print("="*80)
    
    #Check presence of Nan, if low inputation
    for region, x in epigenomes.items():
        print("\n".join((
            f"Nan values report for {region} data:",
            f"In the document there are {x.isna().values.sum()} NaN values out of {x.values.size} values.",
            f"The sample (row) with most values has {x.isna().sum(axis=0).max()} NaN values out of {x.shape[1]} values.",
            f"The feature (column) with most values has {x.isna().sum().max()} NaN values out of {x.shape[0]} values."
        )))
        print("="*80)

    #Knn inputation
    for region, x in epigenomes.items():
        epigenomes[region] = knn_imputer(x)

    bed = to_bed(epigenomes["promoters"])
    print(bed[:5])
    return bed, labels["promoters"]
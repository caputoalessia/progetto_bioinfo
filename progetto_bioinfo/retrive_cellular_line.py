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
from sklearn.preprocessing import RobustScaler
from scipy.stats import pearsonr
from scipy.stats import spearmanr
#from minepy import MINE

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

def robust_zscoring(df:pd.DataFrame)->pd.DataFrame:
    return pd.DataFrame(
        RobustScaler().fit_transform(df.values),
        columns=df.columns,
        index=df.index
    )

def drop_constant_features(df:pd.DataFrame)->pd.DataFrame:
    return df.loc[:, (df != df.iloc[0]).any()]

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

    #promoters_epigenomes = promoters_epigenomes.droplevel(1, axis=1) 
    #enhancers_epigenomes = enhancers_epigenomes.droplevel(1, axis=1)
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

    return epigenomes
    '''
    #Ratio should be > 1
    for region, x in epigenomes.items():
        print(
            f"The rate between features and samples for {region} data is: {x.shape[0]/x.shape[1]}"
        )
        print("="*80)
    
    #Check presence of Nan, if low imputation
    for region, x in epigenomes.items():
        print("\n".join((
            f"Nan values report for {region} data:",
            f"In the document there are {x.isna().values.sum()} NaN values out of {x.values.size} values.",
            f"The sample (row) with most values has {x.isna().sum(axis=0).max()} NaN values out of {x.shape[1]} values.",
            f"The feature (column) with most values has {x.isna().sum().max()} NaN values out of {x.shape[0]} values."
        )))
        print("="*80)

    #Knn imputation
    for region, x in epigenomes.items():
        epigenomes[region] = knn_imputer(x)

    #Gestisco outliars con z-scoring
    epigenomes = {
        region: robust_zscoring(x)
        for region, x in epigenomes.items()
    }

    #Gestiamo features costanti
    for region, x in epigenomes.items():
        result = drop_constant_features(x)
        if x.shape[1] != result.shape[1]:
            print(f"Features in {region} were constant and had to be dropped!")
            epigenomes[region] = result
        else:
            print(f"No constant features were found in {region}!")
    print("="*80)

    #Correlazione lineare con pearson
    p_value_threshold = 0.01
    correlation_threshold = 0.05
    uncorrelated = {
        region: set()
        for region in epigenomes
    }
    for region, x in epigenomes.items():
        for column in tqdm(x.columns, desc=f"Running Pearson test for {region}", dynamic_ncols=True, leave=False):
            correlation, p_value = pearsonr(x[column].values.ravel(), labels[region].values.ravel())
            if p_value > p_value_threshold:
                print(region, column, correlation)
                uncorrelated[region].add(column)
    print("="*80)

    #Correlazione monotona  con Spearman
    for region, x in epigenomes.items():
        for column in tqdm(x.columns, desc=f"Running Spearman test for {region}", dynamic_ncols=True, leave=False):
            correlation, p_value = spearmanr(x[column].values.ravel(), labels[region].values.ravel())
            if p_value > p_value_threshold:
                print(region, column, correlation)
                uncorrelated[region].add(column)
    print("="*80)
    
    #Correlazione non lineare
    for region, x in epigenomes.items():
        for column in tqdm(uncorrelated[region], desc=f"Running MINE test for {region}", dynamic_ncols=True, leave=False):
            mine = MINE()
            mine.compute_score(x[column].values.ravel(), labels[region].values.ravel())
            score = mine.mic()
            if score < correlation_threshold:
                print(region, column, score)
            else:
                uncorrelated[region].remove(column)
    

    print("="*80)
    bed = to_bed(epigenomes["promoters"])
    print(bed[:5])
    return bed, labels["promoters"]
    '''
from progetto_bioinfo.feature_selection import filter_epigenome
from epigenomic_dataset import load_epigenomes
from progetto_bioinfo.correlations import get_correlations
from progetto_bioinfo.retrive_cellular_line import knn_imputer, drop_constant_features, robust_zscoring

def get_data():
    cell_line = "A549"
    window_size = 200
    
    promoters_epigenomes, promoters_labels = load_epigenomes(
        cell_line=cell_line,
        dataset="fantom",
        regions="promoters",
        window_size=window_size
    )

    enhancers_epigenomes, enhancers_labels = load_epigenomes(
        cell_line=cell_line,
        dataset="fantom",
        regions="enhancers",
        window_size=window_size
    )

    promoters_epigenomes = promoters_epigenomes.droplevel(1, axis=1)
    enhancers_epigenomes = enhancers_epigenomes.droplevel(1, axis=1)

    epigenomes = {
        "promoters": promoters_epigenomes,
        "enhancers": enhancers_epigenomes
    }
    labels = {
        "promoters": promoters_labels,
        "enhancers": enhancers_labels
    }

    return epigenomes, labels, cell_line

epigenomes, labels, cell_line = get_data()

def test_knn_imp():
    for region, x in epigenomes.items():
        epigenomes[region] = knn_imputer(x)
    
    for region, x in epigenomes.items():
        assert x.isna().values.sum() == 0

def test_filtering():

    for region, x in epigenomes.items():
        epigenomes[region] = knn_imputer(x)

    filtered_epigenome = filter_epigenome(cell_line, epigenomes, labels)
    assert filtered_epigenome["enhancers"].shape[1] <= epigenomes["enhancers"].shape[1]
    assert filtered_epigenome["promoters"].shape[1] <= epigenomes["promoters"].shape[1]

def test_correlation():
    for region, x in epigenomes.items():
        epigenomes[region] = knn_imputer(x)

    filtered_epigenome = get_correlations(cell_line, epigenomes, labels)
    assert filtered_epigenome["enhancers"].shape[1] <= epigenomes["enhancers"].shape[1]
    assert filtered_epigenome["promoters"].shape[1] <= epigenomes["promoters"].shape[1]

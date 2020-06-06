'''
from epigenomic_dataset import load_epigenomes
from progetto_bioinfo.train_model_epi import train_model_epi
from progetto_bioinfo.train_model_seq import train_model_seq
from progetto_bioinfo.epigenomic_model import FFNN_epi, MLP_epi

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

def test_train_epi():
    epigenomes, labels, cell_line = get_data()
    models = []
    models.append(MLP_epi(epigenomes["promoters"].shape[1]))
    df = train_model_epi(models, epigenomes, labels, "promoters", cell_line, 2, 5)
    assert df.at(0, "loss") > 0
    assert df.at(0, "accuracy") < 1
    assert df.at(0, "loss") < 1
    assert df.at(0, "loss") < 1
'''
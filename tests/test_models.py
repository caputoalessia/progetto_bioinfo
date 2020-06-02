from progetto_bioinfo.epigenomic_model import FFNN_epi, MLP_epi
from progetto_bioinfo.sequences_models import FFNN, CNN

def test_epi_MLP():
    assert MLP_epi(40).name == "MLP"

def test_epi_FFNN():
    assert FFNN_epi(40).name == "FFNN"

def test_seq_FFNN():
    assert FFNN().name == "FFNN"

def test_seq_CNN():
    assert CNN().name == "CNN"
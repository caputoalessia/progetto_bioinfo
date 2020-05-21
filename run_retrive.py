from progetto_bioinfo.retrive_cellular_line import retrive_cell_line
from progetto_bioinfo.correlations import get_correlations
from progetto_bioinfo.visualization import visualize
if __name__ == "__main__":
    epigenomes, labels, sequences = retrive_cell_line("A549", 200)
    epigenomes = get_correlations("A549", epigenomes, labels)
    #visualize("A549", epigenomes, labels, sequences)
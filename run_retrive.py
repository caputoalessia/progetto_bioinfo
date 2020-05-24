from progetto_bioinfo.retrive_cellular_line import retrive_cell_line
from progetto_bioinfo.correlations import get_correlations
from progetto_bioinfo.visualization import visualize
from progetto_bioinfo.feature_selection import filter_epigenome
if __name__ == "__main__":
    cell_line = "A549"
    epigenomes, labels, sequences = retrive_cell_line(cell_line, 200)
    epigenomes = get_correlations(cell_line, epigenomes, labels)
    #visualize(cell_line, epigenomes, labels, sequences)
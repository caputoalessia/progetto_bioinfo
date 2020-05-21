from progetto_bioinfo.retrive_cellular_line import retrive_cell_line
from progetto_bioinfo.correlations import get_correlations
from progetto_bioinfo.visualization import visualize
from progetto_bioinfo.cnn_model import CNN
from progetto_bioinfo.train_model import train_model

from PIL import Image
from glob import glob
from barplots import barplots

if __name__ == "__main__":
    cell_line = "A549"
    epigenomes, labels, sequences = retrive_cell_line(cell_line, 200)
    epigenomes = get_correlations(cell_line, epigenomes, labels)
    #visualize(cell_line, epigenomes, labels, sequences)
    model = CNN()

    df = train_model(model, epigenomes, labels)
    print(df)
    barplots(
        df,
        groupby=["model", "run_type"],
        show_legend=False,
        height=5,
        orientation="horizontal",
        path='barplots/'+ cell_line +'/sequence/{feature}.png',
    )
    for x in glob("barplots/"+ cell_line +"/sequence/*.png"):
        im = Image.open(x)
        im.show()

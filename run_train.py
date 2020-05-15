from progetto_bioinfo.cnn_model import CNN
from progetto_bioinfo.train_model import train_model
from progetto_bioinfo.retrive_cellular_line import retrive_cell_line
from PIL import Image
from glob import glob
from barplots import barplots

if __name__ == "__main__":
    bed, labels = retrive_cell_line("A549", 200)
    model = CNN()
    df = train_model(model, bed, labels)
    print(df)
    barplots(
        df,
        groupby=["model", "run_type"],
        show_legend=False,
        height=5,
        orientation="horizontal",
        path='barplots/sequence/{feature}.png',
    )
    for x in glob("barplots/sequence/*.png"):
        im = Image.open(x)
        im.show()
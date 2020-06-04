from progetto_bioinfo.retrive_cellular_line import retrive_cell_line
from progetto_bioinfo.correlations import get_correlations
from progetto_bioinfo.visualization import visualize
from progetto_bioinfo.sequences_models import CNN, FFNN
from progetto_bioinfo.train_model_seq import train_model_seq
from progetto_bioinfo.train_model_epi import train_model_epi
from progetto_bioinfo.feature_selection import filter_epigenome
from progetto_bioinfo.epigenomic_model import FFNN_epi, MLP_epi


from PIL import Image
from glob import glob
from barplots import barplots
from tqdm.auto import tqdm

if __name__ == "__main__":
    cell_lines = ["A549", "H1"]
    data_type = "sequence"
    regions = ["enhancers", "promoters"]

    for cell_line in tqdm(cell_lines, desc="Prepearing cell line", dynamic_ncols=True):
        epigenomes, labels = retrive_cell_line(cell_line, 200)
        if(data_type == "epigenomic"):
            epigenomes = get_correlations(cell_line, epigenomes, labels)
            epigenomes = filter_epigenome(cell_line, epigenomes, labels)
        #visualize(cell_line, epigenomes, labels)
        for region_type in tqdm(regions, desc="Training new region", dynamic_ncols=True):
            if(data_type == "epigenomic"):
                models = []
                size = epigenomes[region_type].shape[1]
                models.append(MLP_epi(size))
                models.append(FFNN_epi(size))
                df = train_model_epi(models, epigenomes,
                                     labels, region_type, cell_line)
            else:
                models = []
                models.append(CNN())
                models.append(FFNN())
                df = train_model_seq(models, epigenomes,
                                     labels, region_type, cell_line)

            barplots(
                df,
                groupby=["model", "run_type"],
                show_legend=False,
                height=5,
                orientation="horizontal",
                path='barplots/' + cell_line + '/' + region_type +
                '/' + data_type + '/{feature}.png',
            )
            '''
            for x in glob("barplots/" + cell_line +'/'+ region_type +'/'+ data_type +"/*.png"):
                im = Image.open(x)
                im.show()
            '''

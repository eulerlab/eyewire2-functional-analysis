import matplotlib.pyplot as plt
import os

from eyewire2_functional_analysis import data_loader
from eyewire2_functional_analysis.plot import plot_ds_on_morph

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = "plot_DS_on_morph.png"


def main(celltype="ON-OFF DS - ventral"):
    df_rois = data_loader.load_df_rois()
    df = data_loader.load_df_rois_morph(df_rois=df_rois)
    df = df[df['Cell Type'] == celltype].copy()
    df = data_loader.add_skels(df=df, inplace=True)
    df = df[df['skel'].notnull()].copy()
    row = df.sample(n=1, random_state=0).iloc[0]

    fig = plot_ds_on_morph(row)
    fig.savefig(os.path.join(HERE, OUTPUT_FILE), dpi=300)
    plt.show()


if __name__ == "__main__":
    main()

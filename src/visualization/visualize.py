import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
from src.data.data_utils import delete_files


class visualize(object):
    save_image_path = None

    @staticmethod
    def h_bar_plot(value_counts, feature_name):
        value_counts.plot(kind='barh').invert_yaxis()
        plt.title(feature_name)
        plt.tight_layout()
        plt.savefig(f'{visualize.save_image_path}/{feature_name}_h_bar.png')
        plt.clf()
        plt.close()

    @staticmethod
    def hist_plot(data, feature_name):
        data.hist()
        plt.title(feature_name)
        plt.tight_layout()
        plt.savefig(f'{visualize.save_image_path}/{feature_name}_hist.png')
        plt.clf()
        plt.close()

    @staticmethod
    def box_plot(data, feature_name):
        data.plot.box()
        plt.title(feature_name)
        plt.savefig(f'{visualize.save_image_path}/{feature_name}_box_plot.png')
        plt.clf()
        plt.close()

    @staticmethod
    def visualize_data(data, path_to_save):
        """Visualize data Categorical and Numerical.
        """
        visualize.save_image_path = path_to_save
        delete_files(path_to_save)

        for index, value in data.dtypes.iteritems():
            if value == object:  # Categorical data
                visualize.h_bar_plot(data[index].value_counts(), index)
            else:
                visualize.box_plot(data[index], index)
                visualize.hist_plot(data[index], index)

    @staticmethod
    def correlation(data, path_to_save):
        visualize.save_image_path = path_to_save
        plt.figure(figsize=(10, 16))
        corr = data.corr()
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, cmap='summer',
                    vmax=.3,
                    center=0,
                    square=True,
                    linewidths=.5,
                    xticklabels=corr.columns,
                    yticklabels=corr.columns,
                    cbar_kws={"orientation": "horizontal"})

        plt.tight_layout()
        plt.savefig(f'{visualize.save_image_path}.png')
        plt.clf()
        plt.close()

    @staticmethod
    def scatter_matrix(data, path_to_save):
        visualize.save_image_path = path_to_save
        axes = scatter_matrix(data, alpha=0.2)
        plt.figure(figsize=(10, 16))
        for ax in axes.flatten():
            ax.xaxis.label.set_rotation(90)
            ax.yaxis.label.set_rotation(0)
            ax.yaxis.label.set_ha('right')
        plt.tight_layout()
        plt.gcf().subplots_adjust(wspace=0, hspace=0)
        plt.savefig(f'{visualize.save_image_path}.png')
        plt.clf()
        plt.close()

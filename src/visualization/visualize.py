import matplotlib.pyplot as plt
from src.data.data_utils import delete_files


class visualize(object):
    save_image_path = None

    @staticmethod
    def h_bar_plot(value_counts, feature_name):
        fig = plt.figure()
        value_counts.plot(kind='barh').invert_yaxis()
        plt.title(feature_name)
        visualize.save_image(f'{feature_name}_h_bar.png', fig)

    @staticmethod
    def hist_plot(data, feature_name):
        fig = plt.figure()
        data.hist()
        plt.title(feature_name)
        visualize.save_image(f'{feature_name}_hist_plot.png', fig)

    @staticmethod
    def box_plot(data, feature_name):
        fig = plt.figure()
        data.plot.box()
        plt.title(feature_name)
        visualize.save_image(f'{feature_name}_box_plot.png', fig)

    @staticmethod
    def save_image(file_name, fig):
        fig.savefig(f'{visualize.save_image_path}/{file_name}')
        plt.close(fig)

    @staticmethod
    def visualize_data(data, path_to_save):
        """Visualize data Categorical and Numerical.
        """
        visualize.save_image_path = path_to_save
        delete_files(path_to_save)
        print(data.dtypes)

        for index, value in data.dtypes.iteritems():
            if value == object:  # Categorical data
                visualize.h_bar_plot(data[index].value_counts(), index)
            else:
                visualize.box_plot(data[index].describe(), index)
                visualize.hist_plot(data[index], index)

import matplotlib.pyplot as plt


def h_bar_plot(value_counts):
    value_counts.plot(kind='barh').invert_yaxis()
    plt.show()

def hist_plot(column):
    column.hist()
    plt.show()

def box_plot(data):
    data.plot.box()
    plt.show()

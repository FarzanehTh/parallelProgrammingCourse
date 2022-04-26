from subprocess import call
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np
import csv
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def readCSV(file, array_size, time):
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        x = []
        y = []
        for row in reader:
            x.append(int(row[array_size]))
            y.append(float(row[time]))
    return x, y


def drawGraph(x_elements, y_elements, file_name, title, x_lable, y_label, exact):

    new_x = []
    for e in x_elements:
        if e < 1024:
            new_x.append(str(e) + 'B')
        elif 1024 <= e < 1024 * 1024:
            new_x.append(str(int(e / 1024)) + 'KiB')
        elif 1024 * 1024 <= e < 1024 * 1024 * 1024:
            new_x.append(str(int(e / (1024 * 1024))) + 'MiB')
        else:
            new_x.append(str(int(e / (1024 * 1024 * 1024))) + 'GiB')

    plt.plot(new_x, np.array(y_elements), color='darkblue',
             linewidth=3, marker='o', markerfacecolor='wheat', markersize=5)
    plt.ylim(0, max(y_elements) * 2)

    plt.xlabel(x_lable)
    plt.ylabel(y_label)
    plt.title(title)
    ax = plt.gca()
    plt.gcf().autofmt_xdate()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if exact:
        plt.yticks(y_elements)
    plt.savefig(file_name)
    plt.clf()


if __name__ == "__main__":

    x_axis_title = 'Array Size'

    # part 1 - A
    file_name_A_1 = 'memoryBandWidth.csv'
    X1, Y1 = readCSV(file_name_A_1, 'write_size', 'bandwidth')
    drawGraph(X1, Y1, 'memoryBandWidth.pdf', 'Memory BandWidth',
              x_axis_title, 'Bandwidth (bytes/ns)', False)

    X2, Y2 = readCSV(file_name_A_1, 'write_size', 'time')
    drawGraph(X2, Y2, 'memoryWriteLatency.pdf', 'Memory Latency',
              x_axis_title, 'write Latency (ns)', False)

    file_name_A_3 = 'cacheLatency.csv'
    X3, Y3 = readCSV(file_name_A_3, 'array_size', 'Latency')
    drawGraph(X3, Y3, 'cacheLatency.pdf', 'Cache Latency',
              x_axis_title, 'Latency (ns/64bytes)', True)

    # part 1 - B
    file_name_B = 'cache_sizes.csv'
    X4, Y4 = readCSV(file_name_B, 'array_size', 'time')
    drawGraph(X4, Y4, 'cache_sizes.pdf', 'Cache Levels & Latency',
              x_axis_title, 'Time (ns)', False)

    # bonus - cache line size
    file_name_C = 'cache_line.csv'
    X5, Y5 = readCSV(file_name_C, 'array_size', 'time')
    drawGraph(X5, Y5, 'cache_line.pdf', 'Cache Line Size',
              x_axis_title, 'Time (s)', False)

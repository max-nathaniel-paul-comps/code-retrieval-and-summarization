import csv
import sys

import matplotlib.pyplot as plt


def analyze_lengths(file_name, col):
    file_path = "../" + file_name
    file = open(file_path)
    reader = csv.reader(file, delimiter='\t')
    lengths = []
    for row in reader:
        lengths.append(len(row[col].split()))
    print(lengths)

    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=lengths, bins="auto", color='#0504aa', rwidth=.95)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.title('Lengths of desciptions in ' + file_name)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 3:
        analyze_lengths(sys.argv[1], int(sys.argv[2]))
    else:
        analyze_lengths("iyer_csharp/dev.txt", 2)


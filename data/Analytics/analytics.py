import csv
import sys
import matplotlib.pyplot as plt
import statistics as stat


def analyze_lengths(file_name, col, statistics=True):
    file_path = "../" + file_name
    file = open(file_path)
    reader = csv.reader(file, delimiter='\t')
    lengths = []
    for row in reader:
        lengths.append(len(row[col].split()))

    plt.hist(x=lengths, bins="auto", color="blue", rwidth=.95)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.title('Lengths of desciptions in ' + file_name)
    plt.show()

    if statistics:
        print("Mean: " + str(stat.mean(lengths)))
        print("Std Dev: " + str(stat.stdev(lengths)))


if __name__ == "__main__":
    if len(sys.argv) == 3:
        analyze_lengths(sys.argv[1], int(sys.argv[2]))
    else:
        analyze_lengths("iyer_csharp/dev.txt", 2)


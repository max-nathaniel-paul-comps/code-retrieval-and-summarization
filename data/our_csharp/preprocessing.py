import random
import sys
import tqdm
import html
sys.path.append("../../src")
from text_data_utils import *


file = open('data4.csv', encoding='UTF8')
reader = csv.reader(file)
examples = [row for row in reader]
examples = examples[1:]
random.seed(a=420)
random.shuffle(examples)
val_split = int(14 * len(examples) / 16)
test_split = int(15 * len(examples) / 16)


def write_csv_dataset(path, data_rows):
    print("Creating %s" % path)
    new_file = open(path, 'w', encoding='UTF8', newline='')
    writer = csv.writer(new_file)
    for i in tqdm.trange(len(data_rows)):
        title = data_rows[i][0]
        title = html.unescape(repr(title))
        answer = data_rows[i][1]
        code = answer[answer.find('<code>') + 6:answer.find('</code>')]
        code = html.unescape(repr(code))
        writer.writerow([title, code])


write_csv_dataset("train.csv", examples[0: val_split])
write_csv_dataset("val.csv", examples[val_split: test_split])
write_csv_dataset("test.csv", examples[test_split:])


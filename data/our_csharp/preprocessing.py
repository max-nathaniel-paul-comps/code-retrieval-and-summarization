import csv
import sys
sys.path.append("../../src")
from text_data_utils import *


file = open('data3.csv', encoding='UTF8')
reader = csv.reader(file)
newfile = open('processeed_data3.csv', 'w', encoding='UTF8', newline='')
writer = csv.writer(newfile)
rows = 0
for row in reader:
    if rows == 0:
        writer.writerow(row)
        rows += 1
        continue
    # if rows>20:
    #     break
    title = row[0]
    title = preprocess_language(title)
    answer = row[1]
    code = answer[answer.find('<code>')+6:answer.find('</code>')]
    code = preprocess_source_code(code)
    if len(code) > 30 and len(code) < 600:
        writer.writerow([title,code])
        rows +=1

        if rows % 10000 == 0:
            print(rows)

print(rows)

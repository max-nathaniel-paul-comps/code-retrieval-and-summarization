import csv


file = open('data.csv', encoding='UTF8')
reader = csv.reader(file)
newfile = open('processeed_data.csv', 'w')
writer = csv.writer(newfile)
rows = 0
for row in reader:
    if rows == 0:
        writer.writerow(row)
        rows += 1
        continue
    if rows>20:
        break
    new_row = []
    new_row.append(row[0])
    answer = row[1]
    code = answer[answer.find('<code>')+6:answer.find('</code>')]
    if len(code) > 30:
        new_row.append(code)
        writer.writerow(new_row)
        rows +=1

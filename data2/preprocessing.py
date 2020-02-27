import csv


file = open('data2.csv', encoding='UTF8')
reader = csv.reader(file)
newfile = open('processeed_data2.csv', 'w', encoding='UTF8', newline='')
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
    if title[-1] == '?':
        title = title[:-1]
    for opener in ['How do I', 'How do you', 'How can I', 'How to', 'Can I']:
        if title.startswith(opener):
            title = title[len(opener):]
    answer = row[1]
    code = answer[answer.find('<code>')+6:answer.find('</code>')]
    if len(code) > 30 and len(code) < 400:
        writer.writerow([title,code])
        rows +=1

        if rows % 10000 == 0:
            print(rows)

print(rows)

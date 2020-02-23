import sys
sys.path.append("../../src")

from text_data_utils import load_iyer_file

summaries, source_code = load_iyer_file("../iyer_csharp/train.txt")

summaries_file = open("summaries_text.txt", 'w')
for i in range(len(summaries)):
    for j in range(len(summaries[i])):
        summaries_file.write(summaries[i][j] + " ")
    summaries_file.write("\n")

codes_file = open("codes_text.txt", 'w')
for i in range(len(source_code)):
    for j in range(len(source_code[i])):
        codes_file.write(source_code[i][j] + " ")
    codes_file.write("\n")

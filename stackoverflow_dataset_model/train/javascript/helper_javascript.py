import os
import csv
#path = "csharp"
files_lst = os.listdir()

text_lst = []
for file in files_lst:
    if '.txt' in file:
        f = open(file,'r')
        text_lst.append(f.read())
        f.close()


with open("/home/jupyter/nlp/stackoverflow_dataset/train/data_train.csv", mode = "a+") as f:
    write = csv.writer(f)
    for val in text_lst:
        write.writerow([val, 'javascript'])
        





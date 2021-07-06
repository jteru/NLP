import os
import csv

files_lst = os.listdir()

text_lst = []
for file in files_lst:
    if '.txt' in file:
        f = open(file,'r')
        text_lst.append(f.read())
        f.close()


with open("/home/jupyter/nlp/stackoverflow_dataset/test/data_test.csv", mode = "a+") as f:
    write = csv.writer(f)
    for val in text_lst:
        write.writerow([val, 'java'])

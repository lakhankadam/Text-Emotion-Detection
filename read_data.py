import re

file = 'text.txt'
def read_data():
    data = []
    with open(file, 'r')as f:
        for line in f:
            line = line.strip()
            label = ' '.join(line[1:line.find("]")].strip().split())
            text = line[line.find("]")+1:].strip()
            data.append([label, text])
    return data

data = read_data()
print("Number of instances: {}".format(len(data)))
from read_data import read_data
from create_feature import create_feature

def convert_label(item, name): 
    items = list(map(float, item.split()))
    label = ""
    for idx in range(len(items)): 
        if items[idx] == 1: 
            label += name[idx] + " "
    
    return label.strip()

emotions = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]

def format_data():
    data = read_data()
    X_all = []
    y_all = []
    for label, text in data:
        y_all.append(convert_label(label, emotions))
        X_all.append(create_feature(text, nrange=(1, 4)))
    return X_all, y_all
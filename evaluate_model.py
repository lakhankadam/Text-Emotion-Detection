from train_model import train_and_get_models
from read_data import read_data
from store_label import convert_label, emotions
from create_feature import create_feature
from split_data import vectorizer

def print_labels():
    data = read_data()
    l = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]
    l.sort()
    label_freq = {}
    for label, _ in data: 
        label_freq[label] = label_freq.get(label, 0) + 1

    for l in sorted(label_freq, key=label_freq.get, reverse=True):
        print("{:10}({})  {}".format(convert_label(l, emotions), l, label_freq[l]))

def run():
    clifs = train_and_get_models()
    
    emoji_dict = {"joy":"😂", "fear":"😱", "anger":"😠", "sadness":"😢", "disgust":"😒", "shame":"😳", "guilt":"😳"}
    t1 = "This looks so impressive"
    t2 = "I have a fear of dogs"
    t3 = "My dog died yesterday"
    t4 = "I don't love you anymore..!"

    texts = [t1, t2, t3, t4]

    for clf in clifs:
        clf_name = clf.__class__.__name__
        print(clf_name)
        for text in texts:
            features = create_feature(text, nrange=(1, 4))
            features = vectorizer.transform(features)
            prediction = clf.predict(features)[0]
            print(text, emoji_dict[prediction])

if __name__ == '__main__':
    run()
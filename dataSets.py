# label: 0 负面，1 中性，2 正面
import sys
import random
import json
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np
class comment_emotion:
    def __init__(self, vocab_path, text_path):
        with open(text_path, 'r', encoding="utf-8") as fp:
            self.text = json.load(fp)
        with open(vocab_path, 'r', encoding="utf-8") as fp:
            self.vocab = json.load(fp)
    def load_data(self):
        label_ = {0: [], 1: [], 2: []}
        length = 0
        for text in self.text:
            for label, word_list in text.items():
                length = len(word_list) if len(word_list) > length else length
        for text in self.text:
            for label, word_list in text.items():
                la = int(label)
                if len(word_list) == 0:
                    continue
                matrix = []
                for word in word_list:
                    matrix.append(self.vocab[word])
                label_[la].append({
                    'label': la,
                    'matrix': matrix
                })
        random.seed(45)
        leng = sys.maxsize
        train_data = []
        test_data = []
        train_label = []
        test_label = []
        for i, j in label_.items():
            random.shuffle(j)
            leng = len(j) if len(j) < leng else leng
        num_elements = int(leng * 0.8)
        for i, j in label_.items():
            j = list(j)
            train_data.extend([data['matrix'] for data in j[:num_elements]])
            train_label.extend([data['label'] for data in j[:num_elements]])
            test_data.extend([data['matrix'] for data in j[num_elements:leng]])
            test_label.extend([data['label'] for data in j[num_elements:leng]])

        train_data = pad_sequences(train_data, maxlen=length, padding="post", truncating="post", value=0)
        test_data = pad_sequences(test_data, maxlen=length, padding="post", truncating="post", value=0)
        train_label = to_categorical(train_label, num_classes=3)
        test_label = to_categorical(test_label, num_classes=3)

        idx = np.random.permutation(len(train_data))
        train_data = train_data[idx]
        train_label = train_label[idx]
        return train_data, train_label, test_data, test_label

if __name__=='__main__':
    # 检查数据
    comment = comment_emotion('Parameters/vocab.json', 'Parameters/text.json')
    train_data, train_label, test_data, test_label = comment.load_data()
    print(len(train_data), len(train_label), len(test_data), len(test_label))
    print(train_data.shape)
    print(train_label.shape)
    print(test_data.shape)
    print(test_label.shape)

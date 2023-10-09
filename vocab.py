import re
import jieba
import pandas as pd
import json
class vocab:
    def __init__(self, filename):
        self.data_set = pd.read_csv(filename)
        self.text_data = []
        self.vocab = set()
    def generate_data(self):
        pattern = re.compile(r'[\u4e00-\u9fa5a-zA-Z]+')
        for text in self.data_set['正文']:
            text = re.sub(r'回复@.*?:', '', str(text))
            seg_list = jieba.lcut(text)
            words = [word for word in seg_list if re.match(pattern, word)]
            print(words)
            self.text_data.append(words)
            self.vocab.update(words)
    def save(self, vocab_path='./Parameters/vocab.json', text_path='./Parameters/text.json'):
        text_dict = []
        vocab_ = {}
        index_text = 0
        for label in self.data_set['情感属性']:
            if label == '负面':
                la = 0
            elif label == '中性':
                la = 1
            else:
                la = 2
            text_dict.append({
                la: self.text_data[index_text]
            })
            index_text += 1
        for i, word in enumerate(self.vocab):
            vocab_[word] = i+1
        with open(text_path, 'w') as fp:
            json.dump(text_dict, fp)
        with open(vocab_path, 'w') as fp:
            json.dump(vocab_, fp)
if __name__=='__main__':
    vocab = vocab('./comment_emotion_01/评论数据情感标注数据集.csv')
    vocab.generate_data()
    vocab.save('./Parameters/vocab.json', './Parameters/text.json')

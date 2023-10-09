import tensorflow as tf
import json
import jieba
import re
from keras.preprocessing.sequence import pad_sequences
import numpy as np
class vector:
    def __init__(self, vocab_path, dim, model_path):
        with open(vocab_path, 'r') as fp:
            self.vocab = json.load(fp)
        self.dim = dim
        self.model = tf.keras.models.load_model(model_path)
    def vector(self, sentence):
        word_list = []
        for i in sentence:
            pattern = re.compile(r'[\u4e00-\u9fa5a-zA-Z]+')
            text = re.sub(r'回复@.*?:', '', str(i))
            seg_list = jieba.lcut(text)
            words = [self.vocab[word] for word in seg_list if re.match(pattern, word)]
            word_list.append(words)
        return pad_sequences(word_list, maxlen=self.dim, padding="post", truncating="post", value=0)
    def predict_sentence(self, sentence):
        return self.model.predict(self.vector(sentence))
    def predict_list(self, data):
        return self.model.predict(data)
if __name__=='__main__':
    dim = 157
    dict = {0: '负面', 1: '中性', 2: '正面'}
    vec = vector('./Parameters/vocab.json', dim, './model')
    arr = vec.predict_sentence(
            [
                '回复@__你们不知道我是谁:我就觉得奇怪了，你自己都在在美国，还让人滚去美国[二哈]//回复@东教授很忙:滚去你的美国卖菊花乖儿子',
                '回复@诸葛大司马:滚你个美分，去找美爹去吧，不送！//回复@YANGNJ:傻瓜',
                '回复@阿兰198810:自己去查，全世界好多动荡都和他有关。//回复@星醒27:呵呵呵，他在中国没做坏事吧',
                '回复@阿村苦逼机械狗:我朋友现在在那边，他老婆在家照顾即将高考的儿字，高考完一家人都过去，老是问我考不考虑，说在那边有技术，又勤快，在那边好过日子，因为本地人福利好，特懒。//回复@立达淡定:local',
                '增加美国花旗参税对美国种植户影响不了的[给力]美加州气候条件优厚种的参甘苦醇效果顶级 加拿大参跟国内参口感辣苦没啥效果（美国芯片跟国内芯片质量对比一样）增15%税 需求的人还会买美国参[给力]希望商务部多了解美国花旗参效果 别用商务部数据来去制裁一种产品，应制裁美国其它产品才对！[给力]',
                '中国从来都是不挑事也不怕事。我们谁也不怕。坚定的站在祖国一边，跟着中国领导核心走。',
                '回复@社会主义接班人王小明:是你不懂幽默，人家说的是反话//回复@云裳花容露华浓:诶，能说出这样的话，你大概是个蛆吧',
                '回复@陆大大Ol:太强大了，奶粉都抢到外国去了，精英都去拯救美帝了，//回复@基洛夫轰炸你:活在新闻联播里的是你才对',
                '[good]',
                '老胡发视频不言语，这是学偷懒的节奏哦。',
            ]
    )
    print([dict[i] for i in np.argmax(arr, axis=1)])
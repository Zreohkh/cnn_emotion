from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers, losses, optimizers, activations
from dataSets import *
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
class CnnBlock(layers.Layer):
    def __init__(self, num_filters, kernel_size, padding='same', activation='relu'):
        super(CnnBlock, self).__init__()
        self.conv1d = Conv1D(num_filters, kernel_size, padding=padding, activation=activation)
        self.global_max_pooling1d = GlobalMaxPooling1D()

    def call(self, inputs):
        x = self.conv1d(inputs)
        x = self.global_max_pooling1d(x)
        return x

    def get_config(self):
        config = super(CnnBlock, self).get_config()
        layer_config = {'num_filters': self.conv1d.filters,
                        'kernel_size': self.conv1d.kernel_size,
                        'padding': self.conv1d.padding,
                        'activation': activations.serialize(self.conv1d.activation)}
        config.update({'conv1d': serialize(self.conv1d),
                       'global_max_pooling1d': serialize(self.global_max_pooling1d),
                       'layer_config': layer_config})
        return config
class MyModel(Model):
    def __init__(self, size, num_category, kernal_size, out_putdim, num_filters):
        super(MyModel, self).__init__()
        self.embed = layers.Embedding(input_dim=size, output_dim=out_putdim)
        self.conv_layers = []
        for i in range(kernal_size[0], kernal_size[1] + 1):
            self.conv_layers.append(CnnBlock(num_filters, i))
        self.concatenation = Concatenate()
        self.dropout = Dropout(0.6)
        self.dense1 = layers.Dense(units=128, activation='relu')
        self.dense2 = layers.Dense(units=64, activation='relu')
        self.dense3 = layers.Dense(units=32, activation='relu')
        self.output_layer = layers.Dense(units=num_category, activation='softmax')

    def call(self, inputs):
        x = self.embed(inputs)
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_output = conv_layer(x)
            conv_outputs.append(conv_output)
        concate = self.concatenation(conv_outputs)
        droped_concate = self.dropout(concate)
        x = self.dense1(droped_concate)
        x = self.dense2(x)
        x = self.dense3(x)
        outputs = self.output_layer(x)
        return outputs

    def build(self, input_shape):
        super(MyModel, self).build(input_shape)
if __name__=='__main__':
    size = 500000   # 词汇表大小
    maxlen = 157   # 单个样本的最大长度
    num_category = 3 # 类别数
    kernal_size = [2, 6] # 卷积核大小序列
    out_putdim = 20 # 输出维度
    num_filters = 32 # filter个数
    cnn_layers = [] # CnnBlock数

    # 加载数据集
    comment = comment_emotion('Parameters/vocab.json', 'Parameters/text.json')
    x_train, y_train, x_val, y_val = comment.load_data()

    # 定义模型
    model = MyModel(size, num_category, kernal_size, out_putdim, num_filters)
    # 构建模型
    model.build(input_shape=(None, maxlen))
    # 编译模型
    model.compile(loss=losses.CategoricalCrossentropy(),
                  optimizer=optimizers.Adam(),
                  metrics=['accuracy'])

    model.summary()
    # 训练模型
    model.fit(x_train, y_train, epochs=6, batch_size=64, validation_data=(x_val, y_val))
    model.save('./model', save_format='tf')

    test_0 = []
    test_1 = []
    test_2 = []
    test_l0 = []
    test_l1 = []
    test_l2 = []
    for index in range(len(y_val)):
        li = y_val[index]
        if li[0] == 1:
            test_0.append(x_val[index])
            test_l0.append(li)
        elif li[1] == 1:
            test_1.append(x_val[index])
            test_l1.append(li)
        elif li[2] == 1:
            test_2.append(x_val[index])
            test_l2.append(li)
    test_loss, test_acc = model.evaluate(np.array(test_0), np.array(test_l0), verbose=2)
    print('负面 accuracy:', test_acc)
    test_loss, test_acc = model.evaluate(np.array(test_1), np.array(test_l1), verbose=2)
    print('中性 accuracy:', test_acc)
    test_loss, test_acc = model.evaluate(np.array(test_2), np.array(test_l2), verbose=2)
    print('正面 accuracy:', test_acc)
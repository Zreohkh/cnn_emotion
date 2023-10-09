import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from dataSets import *
from predict import vector
import numpy as np
comment = comment_emotion('Parameters/vocab.json', 'Parameters/text.json')
train_data, train_label, test_data, test_label = comment.load_data()
dim = 157
vec = vector('./Parameters/vocab.json', dim, './model')
y_true = np.argmax(test_label, axis=1)
y_pred = np.argmax(vec.predict_list(test_data), axis=1)

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 绘制混淆矩阵
plt.imshow(cm, cmap='Blues')

# 添加标题和标签
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# 显示颜色刻度条
plt.colorbar()

# 设置坐标轴刻度
tick_marks = np.arange(len(np.unique(y_true)))
plt.xticks(tick_marks, np.unique(y_true))
plt.yticks(tick_marks, np.unique(y_true))

# 在像素格中显示数值
thresh = cm.max() / 2.
for i in range(len(np.unique(y_true))):
    for j in range(len(np.unique(y_true))):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

# 调整显示位置
plt.tight_layout()

# 保存图片
plt.savefig('Confusion_Matrix.png')

# 显示图像
plt.show()
#-*-coding:utf-8-*-
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

path = '/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf'
fontprop = matplotlib.font_manager.FontProperties(fname=path)

# 단어 벡터를 분석해볼 임의의 문장들
sentences = ["나 고양이 좋다",
             "나 강아지 좋다",
             "나 동물 좋다",
             "강아지 고양이 동물",
             "여자친구 고양이 강아지 좋다",
             "고양이 생선 우유 좋다",
             "강아지 생선 싫다 우유 좋다",
             "강아지 고양이 눈 좋다",
             "나 여자친구 좋다",
             "여자친구 나 싫다",
             "여자친구 나 영화 책 음악 좋다",
             "나 게임 만화 애니 좋다",
             "고양이 강아지 싫다",
             "강아지 고양이 좋다"]



word_sequence = " ".join(sentences).split()
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
inv_word_dict = {i: w for i, w in enumerate(word_list)}

skip_grams = []

for i in range(1, len(word_sequence) - 1):
    # (context, target) : ([target index - 1, target index + 1], target)
    target = word_dict[word_sequence[i]]
    context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]

    # (target, context[0]), (target, context[1])..
    for w in context:
        skip_grams.append([target, w])





training_epoch = 3000
learning_rate = 0.001
batch_size = 20
embedding_size = 2
num_sampled = 15
voc_size = len(word_list)

print("size data: {}".format(len(skip_grams)))
def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(data[i][0])  # target
        random_labels.append(np.eye(voc_size)[data[i][1]])  # context word

    return random_inputs, random_labels

def getFull(data):
    inputs = []
    labels = []
    for i in range(0,len(data)):
        inputs.append(data[i][0])  # target
        labels.append(np.eye(voc_size)[data[i][1]])  # context word

    return inputs, labels


class W2V(tf.keras.Model):
  def __init__(self):
    super(W2V, self).__init__()
    self.w2v = tf.keras.layers.Embedding(voc_size, embedding_size)
    self.proj = tf.keras.layers.Dense(voc_size,activation='softmax')

  def train(self, x):
    x = self.w2v(x)
    x = self.proj(x)
    return x

  def call(self, x):
    return self.w2v(x)


model = W2V()

def lossFunction(label, pred):
  cost = tf.math.reduce_mean(-tf.math.reduce_sum(label * tf.math.log(pred), axis=1))
  return cost

train_op = tf.keras.optimizers.Adam(learning_rate)


for step in range(1, training_epoch + 1):
  batch_inputs, batch_labels = getFull(skip_grams)
  batch_inputs = np.array(batch_inputs)
  batch_labels = np.array(batch_labels)

  with tf.GradientTape() as tape:
    pred_labels = model.train(batch_inputs)
    loss = lossFunction(batch_labels, pred_labels)
  variables = model.trainable_variables
  gradients = tape.gradient(loss, variables)
  gradients, grad_norm = tf.clip_by_global_norm(gradients, 0.1)
  train_op.apply_gradients(zip(gradients, variables))
  if(step % 50 == 0):
    print("loss: {}".format(loss))



for i, label in enumerate(word_list):
    out  = model(i)
    x, y = out
    plt.scatter(x, y)
    plt.text(x+0.01,y, "{}".format(label), fontsize=10, fontproperties=fontprop)

plt.show()

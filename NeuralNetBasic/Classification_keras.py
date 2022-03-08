# 털과 날개가 있는지 없는지에 따라, 포유류인지 조류인지 분류하는 신경망 모델을 만들어봅니다.
import tensorflow as tf
import numpy as np

# [털, 날개]
x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])

# [기타, 포유류, 조류]
# 다음과 같은 형식을 one-hot 형식의 데이터라고 합니다.
y_data = np.array([
    [1, 0, 0],  # 기타
    [0, 1, 0],  # 포유류
    [0, 0, 1],  # 조류
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

#########
# 신경망 모델 구성
######

class classfication(tf.keras.Model):
  def __init__(self):
    super(classfication, self).__init__()
    self.layer1 = tf.keras.layers.Dense(3, activation='relu')
    self.layer2 = tf.keras.layers.Dense(3, activation='softmax')

  def call(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    return x




# 신경망을 최적화하기 위한 비용 함수를 작성합니다.
# 각 개별 결과에 대한 합을 구한 뒤 평균을 내는 방식을 사용합니다.
# 전체 합이 아닌, 개별 결과를 구한 뒤 평균을 내는 방식을 사용하기 위해 axis 옵션을 사용합니다.
# axis 옵션이 없으면 -1.09 처럼 총합인 스칼라값으로 출력됩니다.
#        Y         model         Y * tf.log(model)   reduce_sum(axis=1)
# 예) [[1 0 0]  [[0.1 0.7 0.2]  -> [[-1.0  0    0]  -> [-1.0, -0.09]
#     [0 1 0]]  [0.2 0.8 0.0]]     [ 0   -0.09 0]]
# 즉, 이것은 예측값과 실제값 사이의 확률 분포의 차이를 비용으로 계산한 것이며,
# 이것을 Cross-Entropy 라고 합니다.

def lossFunction(label, pred):
  cost = tf.math.reduce_mean(-tf.math.reduce_sum(label * tf.math.log(pred), axis=1))
  return cost

optimizer = tf.keras.optimizers.Adam(learning_rate=0.008) 


#########
# 신경망 모델 학습
######

model = classfication() 
for step in range(200):
  with tf.GradientTape() as tape:
    pred = model(x_data)
    loss = lossFunction(y_data, pred)

  variables = model.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))
  if(step % 50 == 0):
    print("loss: {}".format(loss))




#########
# 결과 확인
# 0: 기타 1: 포유류, 2: 조류
######
# tf.argmax: 예측값과 실제값의 행렬에서 tf.argmax 를 이용해 가장 큰 값을 가져옵니다.
# 예) [[0 1 0] [1 0 0]] -> [1 0]
#    [[0.2 0.7 0.1] [0.9 0.1 0.]] -> [1 0]

# [털, 날개]
x_test_data = np.array(
    [[0.1, 0.2], [0.9, 0.1], [0.9, 0.9]])

# [기타, 포유류, 조류]
# 다음과 같은 형식을 one-hot 형식의 데이터라고 합니다.
y_test_data = np.array([
    [1, 0, 0],  # 기타
    [0, 1, 0],  # 포유류
    [0, 0, 1]  # 조류
])


pred = model(x_test_data)
print('예측값: {}'.format(pred))
print('실제값: {}'.format(y_test_data))

pred_label = tf.math.argmax(pred)
real_label = tf.math.argmax(y_test_data)
is_correct = tf.math.equal(pred_label, real_label)
accuracy = tf.math.reduce_mean(tf.cast(is_correct, tf.float32))*100
print('정확도: %.2f' % float(accuracy))






#/*************************************************************************************/
import random
import gym
import numpy as np
from tensorflow.keras import models, layers

env = gym.make("CartPole-v0")  # 加载游戏环境

STATE_DIM, ACTION_DIM = 4, 2  # State 维度 4, Action 维度 2
model = models.Sequential([
    layers.Dense(64, input_dim=STATE_DIM, activation='relu'),
    layers.Dense(40, activation='relu'),
    layers.Dense(ACTION_DIM, activation='linear')
])
model.summary()  # 打印神经网络信息

#/*************************************************************************************/
def generate_data_one_episode():
    x,y,score=[],[],0
    state = env.reset()
    while True:
        action = random.randrange(0,2)
        x.append(state)
        y.append([1,0] if action ==0 else [0,1])
        state ,reward ,done,_ = env.step(action)
        score+=reward
        if done:
            break
    return x,y,score
    #/*************************************************************************************/
    def generate_training_data(expected_score=100):
    data_x,data_y,scores = [],[],[]
    for i in range(10000):
        x,y,score =generate_data_one_episode()
        if score>expected_score:
            data_x += x
            data_y += y
            scores.append(score)
    print('dataset size: {}, max score: {}'.format(len(data_x), max(scores)))
    return np.array(data_x),np.array(data_y)
    #/*************************************************************************************/

data_x ,data_y = generate_training_data()
model.compile(loss='mse',
              optimizer='adam',
              epochs=5
             )
model.fit(data_x,data_y)
model.save('CartPole-v0-nn.h5')
#*************************************************************************************
#train our model 
import time
import numpy as np
import gym
from tensorflow.keras import models


saved_model = models.load_model('CartPole-v0-nn.h5')  # 加载模型
env = gym.make("CartPole-v0")  # 加载游戏环境

for i in range(5):
    state = env.reset()
    score = 0
    while True:
        time.sleep(0.01)
        env.render()   # 显示画面
        action = np.argmax(saved_model.predict(np.array([state]))[0])  # 预测动作
        state, reward, done, _ = env.step(action)  # 执行这个动作
        score += reward     # 每回合的得分
        if done:       # 游戏结束
            print('using nn, score: ', score)  # 打印分数
            break
env.close()


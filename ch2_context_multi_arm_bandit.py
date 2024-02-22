#! -*- coding:utf-8 -*-
import logging
import numpy as np
from scipy import stats
import random
import torch
import matplotlib.pyplot as plt


# Setting logging & logging format
LOGGING_FORMAT = "%(asctime)s %(levelname)s: %(message)s"
DATE_FORMAT = r"%Y%m%d %H:%M:%S"
logging.basicConfig(level = logging.INFO, format = LOGGING_FORMAT, datefmt = DATE_FORMAT)

# plot setting
plt.figure(figsize=(20,7))
plt.ylabel("Average reward",fontsize=14)
plt.xlabel("Training Epochs",fontsize=14)

# ---------- Context multi-arm bandit ---------- #

# ----- 解決情境是拉霸機情境 ----- #
class ContextBandit(): # 拉霸機環境類別
    def __init__(self, arms = 10):
        self.arms = arms # 這裡的 arm 代表廣告
        self.init_distribution(arms)
        self.update_state()

    def init_distribution(self, arms):
        states = arms  # 讓狀態數 = 廣告數以方便處理
        self.bandit_matrix = np.random.rand(states, arms) # 隨機產生的 10 種狀態下的 10 個 arms 的機率分佈（10*10種機率）
    
    def reward(self, prob):
        reward = 0
        for i in range(self.arms):
            if random.random() < prob:
                reward += 1
        return reward

    def update_state(self):
        self.state = np.random.randint(0, self.arms) # 隨機產生一個新狀態

    def get_state(self): # 取得當前狀態
        return self.state

    def get_reward(self, arm):
        return self.reward(self.bandit_matrix[self.get_state()][arm]) # 根據當前狀態及選擇的 arm 傳回回饋值
    
    def choose_arm(self, arm): 
        reward = self.get_reward(arm)
        self.update_state() #產生下一個狀態
        return reward # 傳回回饋值

def one_hot(N, pos, val = 1):
	one_hot_vec = np.zeros(N)
	one_hot_vec[pos] = val
	return one_hot_vec

def softmax(av, tau):
    softm = (np.exp(av / tau) / np.sum(np.exp(av / tau)))
    return softm

def train(arms, env, epochs = 10000, learning_rate = 1e-2):
    N, D_in, H, D_out, = 1, arms, 100, arms

    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H), # 隱藏層
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out), # 輸出層
        torch.nn.ReLU(),
    )
    loss_fn = torch.nn.MSELoss()
    cur_state = torch.Tensor(one_hot(arms, env.get_state())) # 取得環境目前的狀態，並將其編碼為one-hot張量
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    rewards = []
    for i in range(epochs):
        y_pred = model(cur_state) # 執行神經網路並預測回饋值
        av_softmax = softmax(y_pred.data.numpy(), tau=1.12) # 利用 softmax 將預測結果轉換成機率分佈向量
        choice = np.random.choice(arms, p = av_softmax) # 依照softmax輸出的機率分佈來選取新動作
        cur_reward = env.choose_arm(choice) # 執行選擇的動作，並取得一個回饋值
        one_hot_reward = y_pred.data.numpy().copy() # 將資料型別由PyTorch張量轉換成Numpy陣列
        one_hot_reward[choice] = cur_reward # 更新 one_hot_reward 陣列的值，把它當作標籤（實際的回饋值）
        reward = torch.Tensor(one_hot_reward)
        rewards.append(cur_reward) # 將回饋值存入 rewards 中，以便稍後繪製線圖
        loss = loss_fn(y_pred, reward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cur_state = torch.Tensor(one_hot(arms, env.get_state())) # 更新目前的環境狀態
    return np.array(rewards)

def running_mean(x, N =100): #定義一個可以算出移動平均回饋值的函式
    c = x.shape[0] - N
    y = np.zeros(c)
    conv = np.ones(N)
    for i in range(c):
        y[i] = (x[i:i+N] @ conv)/N
    return y

if __name__ == "__main__":
    arms = 10
    env = ContextBandit(arms)
    rewards = train(arms, env) # 開始訓練10000次
    plt.plot(running_mean(rewards, N = 50))
    plt.show()
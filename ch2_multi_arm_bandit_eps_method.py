#! -*- coding:utf-8 -*-
import logging
import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt


# Setting logging & logging format
LOGGING_FORMAT = "%(asctime)s %(levelname)s: %(message)s"
DATE_FORMAT = r"%Y%m%d %H:%M:%S"
logging.basicConfig(level = logging.INFO, format = LOGGING_FORMAT, datefmt = DATE_FORMAT)

# plot setting
fig, ax = plt.subplots(1,1)
ax.set_xlabel("Plays")
ax.set_ylabel("Average Reward")
fig.set_size_inches(9,5)

# ---------- Multi-arm bandit ---------- #

# ----- 利用平均(期望)獎金找出最佳動作 ----- #
def exp_reward(a, history):
    """
    計算拉第 a 機台的平均報酬
    """
    rewards_for_a = history[a]
    return sum(rewards_for_a)/len(rewards_for_a)

def get_best_action(actions, history):
    """
    利用平均(期望)獎金找出最佳動作
    """
    best_action = 0
    max_action_value = 0
    for i in range(len(actions)):
        cur_action_value = exp_reward(actions[i], history)
        if cur_action_value > max_action_value:
            best_action = i # 若cur_action_value比較大，即更新索引best_action的值
            max_action_value = cur_action_value
    return best_action

# ----- 利用 ε 策略解決多臂拉霸機問題 ----- #
def main(probs, eps, record, rewards):
    for i in range(500):
        if random.random() > eps: # 利用（找出平均獎金最高的機台號碼 choice，之前我們叫做 action）
            choice = get_best_arm(record)
        else: # 探索（隨機選出一個機台號碼）
            choice = np.random.randint(10)
        r = get_reward(probs[choice]) # 取得此次遊戲會得到的獎金
        record = update_record(record, choice, r) # 更新record陣列中與該拉霸機號碼對應的遊戲次數和平均獎金
        mean_reward = ((i+1) * rewards[-1] + r)/(i+2) #計算最新的總體平均獎金
        rewards.append(mean_reward) # 記錄到rewards串列
    return rewards

def get_reward(prob): # prob 為某臺拉霸機的中獎率，注意 prob 和之前的 probs 不同，probs是所有拉霸機的中獎率構成的陣列
    """
    把中獎率轉成中獎金額
    """
    reward = 0
    for i in range(10): 	
        if random.random() < prob: # 編註：因為 random() 會產生均勻分佈的亂數，所以在 10 次迴圈中，產生的亂數值小於 prob 的次數會正比於 prob 的大小
            reward += 1 # 若隨機產生的數字小於中獎率，就把 reward 加1
    return reward # 傳回 reward 值(存有本次遊戲中開出的獎金)

def update_record(record, action, r):
    """
    更新 record 內容
    """
    r_ave = (record[action, 0] * record[action, 1] + r) / (record[action, 0] + 1) # 算出新的平均值
    record[action,0] += 1 # action 號機台的拉桿次數加1
    record[action,1] = r_ave # 更新該機台的平均獎金
    return record

def get_best_arm(record):
    """
    找出最佳動作
    """
    arm_index = np.argmax(record[:, 1])
    return arm_index

if __name__ == "__main__":
    n = 10 # 設定拉霸機的數量
    probs = np.random.rand(n) # 隨機設定不同拉霸機的中獎率（0～1之間）
    eps = 0.2 # 設定 ε 為 0.2
    record = np.zeros((10,2)) # 10 個拉霸機個別的拉桿次數(n) 及目前平均金額
    rewards = [0]

    rewards = main(probs, eps, record, rewards)
    ax.scatter(np.arange(len(rewards)), rewards)
    plt.show()



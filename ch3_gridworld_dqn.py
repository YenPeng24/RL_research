#! -*- coding:utf-8 -*-
import logging
import numpy as np
from scipy import stats
import random
import torch
import matplotlib.pyplot as plt
from Gridworld import Gridworld

# Setting logging & logging format
LOGGING_FORMAT = "%(asctime)s %(levelname)s: %(message)s"
DATE_FORMAT = r"%Y%m%d %H:%M:%S"
logging.basicConfig(level = logging.INFO, format = LOGGING_FORMAT, datefmt = DATE_FORMAT)

# ---------- Using DQN in Gridworld ---------- #
class DqnGridworld():
    def __init__(self):
        L1 = 64 # 輸入層的寬度 (4 * 4 * 4), 幀(玩家、終點、陷阱、牆壁), 
        L2 = 150 # 第一隱藏層的寬度
        L3 = 100 # 第二隱藏層的寬度
        L4 = 4 # 輸出層的寬度
        self.model = torch.nn.Sequential(
            torch.nn.Linear(L1, L2), # 第一隱藏層的shape 
            torch.nn.ReLU(),
            torch.nn.Linear(L2, L3), # 第二隱藏層的shape
            torch.nn.ReLU(),
            torch.nn.Linear(L3,L4) # 輸出層的shape
        )
        self.action_set = {
            0: 'u', #『0』代表『向上』
            1: 'd', #『1』代表『向下』
            2: 'l', #『2』代表『向左』
            3: 'r' #『3』代表『向右』
        }
        self.losses = [] # 使用串列將每一次的loss記錄下來，方便之後將loss的變化趨勢畫成圖

    def train(self, epochs = 1000, learning_rate = 1e-3, gamma = 0.9):
        loss_fn = torch.nn.MSELoss() # 指定損失函數為 MSE（均方誤差）
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate) # 指定優化器為Adam，其中model.parameters會傳回所有要優化的權重參數
        epsilon = 1

        for i in range(epochs):
            game = Gridworld(size=4, mode='static')
            state_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10 # 將3階的狀態陣列（4x4x4）轉換成向量（長度為64），並將每個值都加上一些雜訊（很小的數值，避免relu更新有問題）。	
            state1 = torch.from_numpy(state_).float() # 將NumPy陣列轉換成PyTorch張量，並存於state1中
            status = 1 # 用來追蹤遊戲是否仍在繼續（『1』代表仍在繼續）
            while(status == 1):
                qval = self.model(state1) # 執行Q網路，取得所有動作的預測Q值
                qval_ = qval.data.numpy() # 將qval轉換成NumPy陣列
                if (random.random() < epsilon): 
                    action_ = np.random.randint(0,4) # 隨機選擇一個動作（探索）
                else:
                    action_ = np.argmax(qval_) # 選擇Q值最大的動作（探索）        
                action = self.action_set[action_] # 將代表某動作的數字對應到makeMove()的英文字母
                game.makeMove(action) # 執行之前ε—貪婪策略所選出的動作 
                state2_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10
                state2 = torch.from_numpy(state2_).float() # 動作執行完畢，取得遊戲的新狀態並轉換成張量
                reward = game.reward()
                with torch.no_grad(): 
                    newQ = self.model(state2.reshape(1,64))
                maxQ = torch.max(newQ) # 將新狀態下所輸出的Q值向量中的最大值給記錄下來
                if reward == -1:
                    Y = reward + (gamma * maxQ)  # 計算訓練所用的目標Q值
                else: # 若reward不等於-1，代表遊戲已經結束，也就沒有下一個狀態了，因此目標Q值就等於回饋值
                    Y = reward
                Y = torch.Tensor([Y]).detach() 
                X = qval.squeeze()[action_] # 將演算法對執行的動作所預測的Q值存進X，並使用squeeze()將qval中維度為1的階去掉 (shape[1,4]會變成[4])
                loss = loss_fn(X, Y) # 計算目標Q值與預測Q值之間的誤差
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                state1 = state2
                if abs(reward) == 10:       
                    status = 0 # 若 reward 的絕對值為10，代表遊戲已經分出勝負，所以設status為0  
            self.losses.append(loss.item())
            if epsilon > 0.1: 
                epsilon -= (1/epochs) # 讓ε的值隨著訓練的進行而慢慢下降，直到0.1（還是要保留探索的動作）
            if i%100 == 0:
                logging.info(f"Training epochs {i+100} completed.")
    def test(self, mode='static', display=True):
        i = 0
        test_game = Gridworld(size=4, mode=mode) # 產生一場測試遊戲
        state_ = test_game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
        state = torch.from_numpy(state_).float()
        if display:
            logging.info("Initial State:")
            print(test_game.display())
        status = 1
        while(status == 1): #遊戲仍在進行
            qval = self.model(state)
            qval_ = qval.data.numpy()
            action_ = np.argmax(qval_) 
            action = self.action_set[action_]
            if display:
                logging.info(f'Move #: {i}; Taking action: {action}')
            test_game.makeMove(action)
            state_ = test_game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
            state = torch.from_numpy(state_).float()
            if display:
                print(test_game.display())
            reward = test_game.reward()
            if reward != -1: #代表勝利（抵達終點）或落敗（掉入陷阱）
                if reward > 0: #reward>0，代表成功抵達終點
                    status = 2 #將狀態設為2，跳出迴圈
                if display:
                    logging.info(f"Game won! Reward: {reward}")
                else: #掉入陷阱
                    status = 0 #將狀態設為0，跳出迴圈
                    if display:
                        logging.info(f"Game LOST. Reward: {reward}")
            i += 1 #每移動一步，i就加1
            if (i > 15): #若移動了15步，仍未取出勝利，則一樣視為落敗
                if display:
                    logging.info("Game lost; too many moves.")
                break    
        win = True if status == 2 else False
        return win

if __name__ == "__main__":
    dqn = DqnGridworld()

    dqn.train()
    plt.figure(figsize=(10,7))
    plt.plot(dqn.losses)
    plt.xlabel("Epochs",fontsize=11)
    plt.ylabel("Loss",fontsize=11)
    plt.show()

    test_result = dqn.test('static')
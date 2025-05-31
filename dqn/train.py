import numpy as np
import torch
import random
from game import Game
from agent import DQNAgent, ReplayMemory
import time
import matplotlib.pyplot as plt

# 超参
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
LR = 1e-4
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
NUM_EPISODES = 100
MAP_SIZE = 10
N_PLAYERS = 2
MAX_TURNS = 500  # 最大回合数

# 初始化环境和 Agent
env = Game(h=MAP_SIZE, w=MAP_SIZE, n_players=N_PLAYERS)
n_actions = env._get_action_space_size()

agents = [DQNAgent(MAP_SIZE, MAP_SIZE, n_actions, i+1, 
                  gamma=GAMMA, epsilon_start=EPS_START, 
                  epsilon_end=EPS_END, epsilon_decay=EPS_DECAY, 
                  lr=LR, memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE)
          for i in range(N_PLAYERS)]

# 训练循环
episode_rewards = [[] for _ in range(N_PLAYERS)]
win_rates = [0 for _ in range(N_PLAYERS)]

for episode in range(NUM_EPISODES):
    # 重置环境
    state = env.reset()
    states = [env._get_state(i+1) for i in range(N_PLAYERS)]
    dones = [False] * N_PLAYERS
    episode_reward = [0] * N_PLAYERS
    
    # 添加回合控制
    for turn in range(MAX_TURNS):
        if all(dones):
            break
            
        for i in range(N_PLAYERS):
            if dones[i]:
                continue
                
            # 选择动作
            action_tuple = agents[i].select_action(states[i], env)
            
            # 编码动作
            action_encoded = env._encode_action(*action_tuple)
            
            # 执行动作
            next_state, reward, done, info = env.step(i+1, action_tuple)
            
            # 存储经验
            agents[i].memory.push(
                states[i], 
                action_encoded,
                next_state, 
                reward, 
                done
            )
            
            states[i] = next_state
            episode_reward[i] += reward
            dones[i] = done
            agents[i].optimize_model()
        
        # 下一个 Turn
        env.update()
    
    # 更新目标网络
    if episode % TARGET_UPDATE == 0:
        for agent in agents:
            agent.update_target_net()
    
    # 记录结果
    for i in range(N_PLAYERS):
        episode_rewards[i].append(episode_reward[i])
        if dones[i] and env.done[i+1]:  # 这个玩家是胜利者
            win_rates[i] += 1
    
    # 打印进度
    # if episode % 10 == 0:
    print(f'Episode {episode}, Win rates: {[w/(episode+1) for w in win_rates]}, Avg rewards: {[np.mean(r[-10:]) if len(r)>=10 else np.mean(r) for r in episode_rewards]}')

# 保存模型
for i, agent in enumerate(agents):
    agent.save_model(f'dqn_agent_{i+1}.pth')

# 绘制训练曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for i in range(N_PLAYERS):
    plt.plot(episode_rewards[i], label=f'Player {i+1}')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()

plt.subplot(1, 2, 2)
for i in range(N_PLAYERS):
    plt.plot(np.cumsum([1 if r > 0 else 0 for r in episode_rewards[i]]) / np.arange(1, NUM_EPISODES+1), 
             label=f'Player {i+1}')
plt.xlabel('Episode')
plt.ylabel('Win Rate')
plt.legend()
plt.tight_layout()
plt.savefig('training_results.png')
plt.show()
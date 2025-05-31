import random
import numpy as np
from collections import namedtuple, deque
import torch
import torch.optim as optim
from net import DQN
import torch.nn.functional as F
from game import Game
from timing import timer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                       ('state', 'action', 'next_state', 'reward', 'done')) # Replay 单元


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity) # 缓冲区
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, h, w, n_actions, player_id, 
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, 
                 epsilon_decay=0.995, lr=1e-4, memory_size=10000, batch_size=32):
        self.h = h
        self.w = w
        self.n_actions = n_actions # 动作空间大小
        self.player_id = player_id
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        self.policy_net = DQN(h, w, n_actions).to(device)
        self.target_net = DQN(h, w, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_size)
        self.steps_done = 0
    
    @timer
    def select_action(self, state, game):
        sample = random.random()
        self.steps_done += 1
        if sample > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.max(1)[1].view(1, 1).item()
                source_pos, direction = game._decode_action(action_idx)
                
                # 验证动作是否有效
                if (game.map[source_pos[0], source_pos[1], 2] == self.player_id 
                    and game.map[source_pos[0], source_pos[1], 1] > 1.5
                    and not self._is_invalid_direction(game, source_pos, direction)):
                    return (source_pos, direction)
        
        # 获取所有玩家领土位置
        player_positions = np.argwhere(
            (game.map[:, :, 2] == self.player_id) & 
            (game.map[:, :, 1] > 1.5)
        )
        
        if len(player_positions) == 0:
            # 没有可用领土，返回任意位置
            source_pos = (np.random.randint(0, self.h), np.random.randint(0, self.w))
            direction = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
            return (source_pos, direction)
        
        # 随机选择玩家领土位置
        source_pos = tuple(player_positions[np.random.randint(0, len(player_positions))])
        
        # 优先有效方向
        valid_directions = []
        for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            if not self._is_invalid_direction(game, source_pos, direction):
                valid_directions.append(direction)

        if valid_directions:
            direction = random.choice(valid_directions)
        else:
            direction = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
        
        return (source_pos, direction)
    
    def _is_invalid_direction(self, game, source_pos, direction):
        """检查方向是否无效（撞山或超出边界）"""
        target_pos = (source_pos[0] + direction[0], source_pos[1] + direction[1])
        
        if (target_pos[0] < 0 or target_pos[0] >= game.h or 
            target_pos[1] < 0 or target_pos[1] >= game.w):
            return True
        if game.map[target_pos[0], target_pos[1], 0] == 2:
            return True
        
        return False
    
    @timer
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda d: not d, batch.done)), 
                                    device=device, dtype=torch.bool)
        
        state_batch_np = np.array(batch.state)
        state_batch = torch.FloatTensor(state_batch_np).to(device)
        
        next_states = np.array(batch.next_state)
        non_final_mask_np = ~np.array(batch.done)
        if np.any(non_final_mask_np):
            non_final_next_states = torch.FloatTensor(next_states[non_final_mask_np]).to(device)
        else:
            non_final_next_states = None
        
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(batch.reward).to(device)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(self.batch_size, device=device)
        if non_final_next_states is not None:
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # 衰减epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    @timer
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, path):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
import numpy as np
from pynoise.noisemodule import *
from pynoise.noiseutil import *
from matplotlib import pyplot as plt
import time
import cv2
from timing import timer

class Game:
    def __init__(self, h, w, p_mountain=0.5, p_city=0.1, n_players=2, is_preview=True):
        '''创建环境对象
        
        Args:
            h: 地图高度
            w: 地图宽度
            p_mountain: 生成山的概率，默认为 0.5
            p_city: 生成城市的概率，默认为 0.1
            n_players: 玩家数量

        Returns:
            环境对象
        '''
        self.h = h
        self.w = w
        self.n_players = n_players
        self.p_mountain = p_mountain
        self.p_city = p_city
        self.map = self.reset()
        if is_preview:
            self.is_preview = True
            output = f'./dqn_{round(time.time())}.mp4'
            height = 100
            weight = 100
            fps = 60
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output, fourcc, fps, (weight, height))
    
    @timer
    def reset(self):
        '''重置环境

        Returns:
            初始 map，形状为 `(h, w, 3)`
            - 第 0 维: 格子类型, 4 将军, 3 城市, 2 山, 1 玩家领土, 0 空
            - 第 1 维: 格子兵力数
            - 第 2 维: 占有此处的玩家 ID [1, n_players], 0 表示空地或山
        '''
        seed = int(time.time() * 1000) % 998244353
        np.random.seed(seed)
        perlin = Perlin(seed=seed)
        noise_map = noise_map_plane(self.w, self.h, 2, 6, 1, 5, perlin)
        noise_map = noise_map.reshape((self.h, self.w))

        map = np.zeros((self.h, self.w, 3), dtype=int)
        # 原谅这里使用 for loop 但只在初始化时运行一次，负担应当不会太大
        for i in range(0, self.h):
            for j in range(0, self.w):
                if noise_map[i, j] > 1 - self.p_mountain or np.random.uniform() < self.p_mountain / 50:
                    map[i, j, 0] = 2
                if np.random.uniform() < (self.p_city + 0.1) / 10:
                    map[i, j, 0] = 3
                    map[i, j, 1] = np.random.randint(40, 50)

        y = np.random.randint(0, self.h, self.n_players)
        x = np.random.randint(0, self.w, self.n_players)
        for i in range(0, self.n_players):
            map[y[i], x[i], 0] = 4
            map[y[i], x[i], 1] = 1
            map[y[i], x[i], 2] = i + 1
        
        self.map = map
        self.turn = 0
        self.done = np.ones((self.n_players+1,), dtype=bool)
        return self.map

    @timer
    def preview(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = np.repeat(np.repeat(255 * (self.map[:, :, 2] == 1), 10, axis=0), 10, axis=1)
        img[:, :, 1] = np.repeat(np.repeat(255 * (self.map[:, :, 2] == 2), 10, axis=0), 10, axis=1)
        self.video_writer.write(img)
    
    @timer
    def update(self):
        '''更新 Turn'''
        if self.turn % 25 == 0: # 领土全部 + 1
            self.map[self.map[:, :, 0] == 1, 1] += 1
        self.map[(self.map[:, :, 0] == 3) | (self.map[:, :, 0] == 4), 1] += 1 # 城市和将军 + 1
        self.turn += 1
        if self.is_preview:
            self.preview()

    # source_pos: (h, w)
    # direction: (0, 1) or (0, -1) or (1, 0) or (-1, 0)
    @timer
    def move(self, player_id, source_pos, direction):
        '''某个玩家 i 进行一步操作

        Args:
            player_id: 玩家 ID，从 `1` 开始到 `n_players`
            source_pos: 源位置 `(h, w)`
            direction: 移动方向 `(0, 1)` or `(0, -1)` or `(1, 0)` or `(-1, 0)`
        
        Returns:
            如果移动是错误的(撞山或超出边界)，返回 `None`

            如果移动成功，返回一个列表，包含两个元素，第一个元素是移动的结果类型，第二个元素是目标位置的结果兵力数

            结果类型
            - 1: 兵力集聚
            - 2: 攻陷敌人领土
            - 3: 拓广空地
            - 4: 攻陷敌人将军
            - 5: 攻陷敌人城市
        '''
        final_return = [0, 0]
        # 源位置是自己家，源兵力至少是2
        if self.map[source_pos[0], source_pos[1], 2] == player_id and self.map[source_pos[0], source_pos[1], 1] > 1.5:
            target_pos = (source_pos[0] + direction[0], source_pos[1] + direction[1])
        else:
            return None
        
        # 没有超出边界，没有撞山
        if self.map[target_pos[0], target_pos[1], 0] == 2 or target_pos[0] < 0 or target_pos[0] >= self.h\
            or target_pos[1] < 0 or target_pos[1] >= self.w:
            return None
        else:
            moving = self.map[source_pos[0], source_pos[1], 1] - 1
            self.map[source_pos[0], source_pos[1], 1] = 1

            if self.map[target_pos[0], target_pos[1], 2] == player_id: # 是自己家
                self.map[target_pos[0], target_pos[1], 1] += moving
                final_return = [1, self.map[target_pos[0], target_pos[1], 1]]

            else: # 不是自己家
                self.map[target_pos[0], target_pos[1], 1] -= moving
                if self.map[target_pos[0], target_pos[1], 1] < 0:
                    self.map[target_pos[0], target_pos[1], 2] = player_id # 换主
                    self.map[target_pos[0], target_pos[1], 1] = -self.map[target_pos[0], target_pos[1], 1] # target = (source - 1) - target
                    if self.map[target_pos[0], target_pos[1], 0] == 4: # 攻陷的是将军
                        self.map[target_pos[0], target_pos[1], 0] = 3 # 变成城市
                        self.done[self.map[target_pos[0], target_pos[1], 2]] = False # 这个位置的玩家输了
                        final_return = [4, self.map[target_pos[0], target_pos[1], 1]]

                    elif self.map[target_pos[0], target_pos[1], 0] == 0: # 攻陷的是空地
                        self.map[target_pos[0], target_pos[1], 0] = 1 # 变成玩家领土
                        final_return = [3, self.map[target_pos[0], target_pos[1], 1]]

                    elif self.map[target_pos[0], target_pos[1], 0] == 1: # 攻陷的是敌人领土
                        final_return = [2, self.map[target_pos[0], target_pos[1], 1]]

                    elif self.map[target_pos[0], target_pos[1], 0] == 3: # 攻陷的是敌人城市
                        final_return = [5, self.map[target_pos[0], target_pos[1], 1]]
                else: # 刚好平了
                    final_return = [1, self.map[target_pos[0], target_pos[1], 1]]

        return final_return
    
    def _get_state(self, player_id):
        '''获取玩家 `player_id` 的状态表示
        
        Args:
            player_id: 玩家 ID，从 `1` 开始到 `n_players`
        
        Returns:
            state: 状态表示，形状为 `(h, w, 7)`
            - 第 0 维: 玩家领土二值掩码
            - 第 1 维: 敌方领土二值掩码
            - 第 2 维: 空地二值掩码
            - 第 3 维: 山地二值掩码
            - 第 4 维: 玩家兵力分布归一化
            - 第 5 维: 敌方兵力分布归一化
            - 第 6 维: 城市二值掩码
            - 第 7 维: 将军二值掩码
        '''

        state = np.zeros((self.h, self.w, 8), dtype=np.float32)
        state[:,:,0] = (self.map[:,:,2] == player_id).astype(float) # 玩家领土二值掩码
        state[:,:,1] = ((self.map[:,:,2] > 0) & (self.map[:,:,2] != player_id)).astype(float) # 敌方领土二值掩码
        state[:,:,2] = (self.map[:,:,0] == 0).astype(float) # 空地二值掩码
        state[:,:,3] = (self.map[:,:,0] == 2).astype(float) # 山地二值掩码

        player_mask = (self.map[:,:,2] == player_id)
        max_army = np.max(self.map[:,:,1]) if np.max(self.map[:,:,1]) > 0 else 1
        state[:,:,4][player_mask] = self.map[:,:,1][player_mask] / max_army # 玩家兵力分布归一化
        
        enemy_mask = ((self.map[:,:,2] > 0) & (self.map[:,:,2] != player_id))
        state[:,:,5][enemy_mask] = self.map[:,:,1][enemy_mask] / max_army # 敌方兵力分布归一化

        state[:,:,6] = (self.map[:,:,0] == 3).astype(float) # 城市二值掩码
        state[:,:,7] = (self.map[:,:,0] == 4).astype(float) # 将军二值掩码
        
        return state
    
    @timer
    def step(self, player_id, action):
        '''智能体 `player_id` 进行一次交互
        
        Args:
            player_id: 玩家 ID，从 `1` 开始到 `n_players`
            action: 一个元组，包含源位置 `(h, w)` 和移动方向 `(0, 1)` or `(0, -1)` or `(1, 0)` or `(-1, 0)`

        Returns:
            - new_state 更新后的状态，形状为 `(h, w, 3)`
            - reward 本次交互的奖励
            - done 游戏是否结束
            - success 本次交互的 info，如果移动成功，包含一个列表，包含两个元素，第一个元素是移动的结果类型，第二个元素是目标位置的结果兵力数
        '''
        # action: (source_pos, direction)
        source_pos, direction = action
        success = self.move(player_id, source_pos, direction)
        reward = self._calculate_reward(player_id, success)
        done = self._check_done()
        new_state = self._get_state(player_id)
        return new_state, reward, done, success
    
    @timer
    def _encode_action(self, source_pos, direction):
        direction_idx = {
            (0, 1): 0,  # 右
            (1, 0): 1,  # 下
            (0, -1): 2, # 左
            (-1, 0): 3  # 上
        }[direction]
        return source_pos[0] * self.w * 4 + source_pos[1] * 4 + direction_idx
    
    @timer
    def _decode_action(self, action_idx):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 右、下、左、上
        direction_idx = action_idx % 4
        source_idx = action_idx // 4
        source_row = source_idx // self.w
        source_col = source_idx % self.w
        return (source_row, source_col), directions[direction_idx]
    
    @timer
    def _get_action_space_size(self):
        return self.h * self.w * 4

    @timer
    def _calculate_reward(self, player_id, success):
        '''计算玩家 `player_id` 的奖励'''
        reward = 0
        if success:
            reward += {
                1: 0.1,  # 兵力集聚
                2: 0.2,  # 攻陷敌人领土
                3: 0.1,  # 拓广空地
                4: 3.0,  # 攻陷敌人将军
                5: 1.0   # 攻陷敌人城市
            }[success[0]]
        else:
            reward += -2.0

        if not self.done[player_id]: # 输了
            reward += -10
        elif self._check_done(): # 赢了
            reward += 10
        return reward
    
    @timer
    def _check_done(self):
        '''检查游戏是否结束
        
        Returns:
            bool: 游戏是否结束
        '''
        return np.sum(self.done[1:]) == 1
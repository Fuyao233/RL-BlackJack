"""
21点环境，从 OpenAI Gym 库修改而来
增加庄家策略功能，环境初始化时可以指定庄家策略
=======
state: [玩家手牌列表], 庄家明牌，是否有可用的 Ace
action: {要牌(1)，停牌(0)}
reward: 胜利 1，失败 -1，平局 0
"""
import sys
import random
import pickle

class BasePolicy:
    """策略基类
    可以用这个基类派生出自己的策略类，当然策略类也可以自己写
    需要注意的是，提交的策略类必须有一个输入观测状态输出动作的act函数，其中观测状态是一个三维元组，三个维度依次是
    自己的手牌和、对手亮出的一张牌、自己有没有可以当作11点使用的Ace牌（详细情况参考下面的get_obs函数）
    """
    def __init__(self) -> None:
        self.foo = None

    def act(self, obs):
        raise NotImplementedError('Policy.act Not Implemented')

    def save(self, path):
        """示例参数保存方法
        """
        # 保存参数
        pickle.dump(self.foo, open(path, 'wb'))

    def load(self, path):
        """示例参数读取方法
        """
        # 读取参数
        self.foo = pickle.load(open(path, 'rb'))


class SimplePolicy:
    def __init__(self, threshold=17) -> None:
        """基本策略

        Parameters
        ----------
        threshold : int, optional
            手牌总点数小于这个阈值时要牌, by default 17
        """
        self.threshold = threshold

    def act(self, obs):
        return 1 if obs[0] < self.threshold else 0


def cmp(a, b):
    return float(a > b) - float(a < b)

# 是否有可用的 Ace
def usable_ace(hand):  
    return 1 in hand and sum(hand) <= 11

# 返回当前手牌的和
def sum_hand(hand):  
    return sum(hand) + 10 if usable_ace(hand) else sum(hand)

# 是否爆牌
def is_bust(hand):
    return sum_hand(hand) > 21

def score(hand):
    """手牌的分数

    Args:
        hand (list): 手牌列表

    Returns:
        int: 得分，如果爆牌返回0，否则返回手牌的点数和
    """
    return 0 if is_bust(hand) else sum_hand(hand)

# 是否天和
def is_natural(hand):
    return sorted(hand) == [1, 10]

class BlackJack:
    def __init__(self, dealer_policy, logger=None) -> None:
        super().__init__()

        # 动作空间
        # 1: 要牌
        # 0: 停牌
        self.action_space = (0, 1)

        # 游戏状态:
        # 0 玩家抽卡阶段，玩家停止抽卡时进入下一阶段
        # 1 庄家抽卡阶段
        # 2 结算阶段
        self.state = 0

        # 牌堆：Ace = 1，2-10 为对应数字，J/Q/K = 10
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

        # 玩家与庄家的卡
        self.player_trajectory = []
        self.dealer_trajectory = []

        # 庄家策略
        self.dealer_policy = dealer_policy

        # 日志
        self.logger = logger or sys.stdout

    def reset(self, seed=None):
        """重置环境，设置随机数种子

        Args:
            seed (int, optional): 随机数种子. Defaults to None.

        Returns:
            [int], int, bool: 观测值（玩家手牌，庄家的名牌，是否有可用的Ace）
        """
        if seed:
            random.seed(seed)
        
        # 给每人发两张牌
        self.player_trajectory = [self._get_card(), self._get_card()]
        self.dealer_trajectory = [self._get_card(), self._get_card()]

        self.state = 0
        # self.logger.write(f'玩家手牌：{self.player_trajectory}，庄家手牌：{self.dealer_trajectory}\n')

        return self._get_obs()

    def step(self, action):
        assert action in self.action_space
        if action:  # 玩家抽卡
            # 抽卡
            self.player_trajectory.append(self._get_card())
            # self.logger.write(f'玩家抽牌，得到：{self.player_trajectory}\n')

            # 检测是否爆牌
            if is_bust(self.player_trajectory):  # 爆牌
                # self.logger.write(f'玩家爆牌，庄家胜利\n')
                return self._get_obs(), -1, True, {}

            return self._get_obs(), 0, False, {}
        
        else: # 玩家停止要牌
            # 进入庄家决策阶段
            self.state += 1
            # self.logger.write('玩家停牌，进入庄家阶段\n')
            # self.logger.write(f'庄家手牌：{self.dealer_trajectory}\n')

            # 获取庄家观测
            dealer_obs = self._get_obs(role='dealer')
            action = self.dealer_policy.act(dealer_obs)
            while action == 1:
                # 庄家抽排
                self.dealer_trajectory.append(self._get_card())
                dealer_obs = self._get_obs(role='dealer')
                # self.logger.write(f'庄家抽牌，得到：{self.dealer_trajectory}\n')

                # 爆牌检测，如果庄家爆牌，玩家得到1的回报
                if is_bust(self.dealer_trajectory):
                    # self.logger.write(f'庄家爆牌，玩家胜利\n')
                    return self._get_obs(), 1, True, {}
                
                action = self.dealer_policy.act(dealer_obs)

            # 庄家停止要牌，开始结算
            # self.logger.write('庄家停牌，开始结算\n')
            self.state += 1
            player_point = score(self.player_trajectory)
            dealer_point = score(self.dealer_trajectory)
            # self.logger.write(f'玩家点数: {player_point}：{self.player_trajectory}\n')
            # self.logger.write(f'庄家点数: {dealer_point}：{self.dealer_trajectory}\n')
            if player_point > dealer_point:
                reward = 1
                # self.logger.write('玩家胜利\n')
            elif player_point == dealer_point:
                reward = 0
                # self.logger.write('平局\n')
            else:
                reward = -1
                # self.logger.write('庄家胜利\n')
            return self._get_obs(), reward, True, {}

    def _get_card(self):
        return int(random.choice(self.deck))

    def _get_obs(self, role='player'):
        """
        获取观测

        Returns
        -------
        int, int, bool: 玩家手牌之和，庄家明牌，玩家是否有可用的Ace
        """
        if role == 'player':
            return (sum_hand(self.player_trajectory), self.dealer_trajectory[0], usable_ace(self.player_trajectory))
        else:
            return (sum_hand(self.dealer_trajectory), self.player_trajectory[0], usable_ace(self.dealer_trajectory))



if __name__ == '__main__':
    env = BlackJack(dealer_policy=SimplePolicy(17))
    print('='*120)
    print('玩家阶段')
    obs = env.reset()
    info = f'玩家手牌{obs[0]:3}, 庄家明牌 {obs[1]:3}，是否有可用Ace：{obs[2]:2}:玩家决策：'
    x = int(input(info))
    while x in {0, 1}:
        obs, reward, done, _ = env.step(x)

        if done:
            if reward == 1:
                print('玩家胜利')
            elif reward == 0:
                print('平局')
            else:
                print('庄家胜利')
            print('='*120)
            obs = env.reset()
            print('玩家阶段')

        # info = f'玩家手牌{obs[0]:3}, 庄家明牌 {obs[1]}，是否有可用Ace：{obs[2]:2}:玩家决策：'
        # x = int(input(info))
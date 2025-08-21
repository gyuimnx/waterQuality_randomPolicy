#train.py
import numpy as np
import matplotlib.pyplot as plt

from env import WaterParkEnv
from agent import QAgent, FixedIntervalPolicy, quantize_state

def run_policy_full(env, policy, quantize=False, episodes=5000):
    total_rewards, usage_counts, safeties = [], [], []
    for ep in range(episodes):
        state = env.reset()
        rewards = 0
        done = False
        safe = True
        while not done:
            s = quantize_state(state) if quantize else state
            action = policy.choose_action(s)
            state, reward, done, info = env.step(action)
            rewards += reward
            if state[0] > 0.5 or state[1] > 2.8 or state[2] < 5.8 or state[2] > 8.6:
                safe = False
        # #에피소드 종료 후 남은 자원 기반 보너스
        # bonus = 0.2 * state[3]   #state[3] = remaining_ci
        # rewards += bonus
        if env.usedCI_count > env.max_ci:  #--------------------------------------------수정
            penalty = (env.usedCI_count - env.max_ci) * 5.0
            rewards -= penalty
        
        total_rewards.append(rewards)
        usage_counts.append(env.usedCI_count)
        safeties.append(safe)
    return total_rewards, usage_counts, safeties

def train_qlearning_full(env, agent, episodes=5000):
    rewards, usages, safeties = [], [], []
    for ep in range(episodes):
        state = env.reset()
        state_disc = quantize_state(state)
        done = False
        total_reward = 0
        safe = True
        while not done:
            action = agent.choose_action(state_disc)
            next_state, reward, done, info = env.step(action)
            next_state_disc = quantize_state(next_state)
            agent.learn(state_disc, action, reward, next_state_disc)
            state_disc = next_state_disc
            state = next_state
            total_reward += reward
            if state[0] > 0.5 or state[1] > 2.8 or state[2] < 5.8 or state[2] > 8.6:
                safe = False
        # #에피소드 종료 후 남은 자원 기반 보너스
        # bonus = 0.2 * state[3] #조정 필요
        # total_reward += bonus
        if env.usedCI_count > env.max_ci:  #--------------------------------------------수정
            penalty = (env.usedCI_count - env.max_ci) * 5.0
            total_reward -= penalty
            
        rewards.append(total_reward)
        usages.append(env.usedCI_count)
        safeties.append(safe)
        agent.decay_epsilon()
        if (ep+1) % 100 == 0:
            print(f"Episode {ep+1} / Epsilon: {agent.epsilon:.4f}")
    return rewards, usages, safeties

def moving_average(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode='valid')

#E-greedy 정책 클래스(일단 삭제)
# class EpsilonGreedyQPolicy:
#     def __init__(self, q_agent, epsilon=0.01):
#         self.q_agent = q_agent
#         self.epsilon = epsilon

#     def choose_action(self, state):
#         if np.random.rand() < self.epsilon:
#             return np.random.randint(self.q_agent.n_actions)
#         return np.argmax(self.q_agent.Q_table[state])

if __name__ == "__main__":
    env = WaterParkEnv()

    #Q-러닝 학습, epsilon 0.1로 시작, decay로 점차 감소
    q_agent = QAgent(epsilon=0.3, epsilon_decay=0.999, epsilon_min=0.001)
    fixed_policy = FixedIntervalPolicy()

    #Fixed Policy
    fixed_rewards, fixed_usage, fixed_safety = run_policy_full(env, fixed_policy, quantize=False, episodes=10000)

    #Q-Learning
    q_rewards, q_usage, q_safety = train_qlearning_full(env, q_agent, episodes=10000)

    #Epsilon-Greedy Policy 평가(epsilon=0.01)
    # greedy_policy = EpsilonGreedyQPolicy(q_agent, epsilon=0.01)
    # greedy_rewards, greedy_usage, greedy_safety = run_policy_full(env, greedy_policy, quantize=True, episodes=10000)

    plt.figure(figsize=(14, 5))

    #전체 리워드(왼쪽)
    plt.subplot(1, 2, 1)
    plt.plot(moving_average(fixed_rewards), label="Fixed Policy")
    plt.plot(moving_average(q_rewards), label="Q-Learning")
    # plt.plot(moving_average(greedy_rewards), label="Greedy Policy")
    plt.title("Policy Performance Comparison")
    plt.ylabel("Mean Total Reward (Moving Average)")
    plt.legend()

    #자원 소모량(오른쪽)
    plt.subplot(1, 2, 2)
    plt.plot(moving_average(fixed_usage), label="Fixed Policy")
    plt.plot(moving_average(q_usage), label="Q-Learning")
    # plt.plot(moving_average(greedy_usage), label="Greedy Policy")
    plt.title("Resource Usage Comparison")
    plt.ylabel("Chlorine Usage (kg, Moving Average)")

    plt.tight_layout()
    plt.show()

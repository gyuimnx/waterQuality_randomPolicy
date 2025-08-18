import numpy as np
import matplotlib.pyplot as plt

from env_waterpark import WaterParkEnv
from agent_waterpark import QAgent, FixedIntervalPolicy, quantize_state

def run_policy_full(env, policy, quantize=False, episodes=5000):
    total_rewards, replace_counts, safeties = [], [], []
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
        # 에피소드 종료 후 남은 교체 횟수에 따른 추가 보상
        bonus = 0.2 * state[3]
        rewards += bonus
        total_rewards.append(rewards)
        replace_counts.append(env.replace_count)
        safeties.append(safe)
    return total_rewards, replace_counts, safeties

def train_qlearning_full(env, agent, episodes=5000):
    rewards, replaces, safeties = [], [], []
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
        # 에피소드 종료 후 남은 교체 횟수에 따른 추가 보상
        bonus = 0.2 * state[3]
        total_reward += bonus
        rewards.append(total_reward)
        replaces.append(env.replace_count)
        safeties.append(safe)
        agent.decay_epsilon()
        if (ep+1) % 100 == 0:
            print(f"Episode {ep+1} / Epsilon: {agent.epsilon:.4f}")
    return rewards, replaces, safeties

def moving_average(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode='valid')

if __name__ == "__main__":
    env = WaterParkEnv()

    # Q-러닝 학습, epsilon 0.1로 시작, decay로 점차 감소
    q_agent = QAgent(epsilon=0.1, epsilon_decay=0.9995, epsilon_min=0.001)
    fixed_policy = FixedIntervalPolicy()

    # Fixed Policy
    fixed_rewards, fixed_replace, fixed_safety = run_policy_full(env, fixed_policy, quantize=False, episodes=10000)

    # Q-Learning
    q_rewards, q_replace, q_safety = train_qlearning_full(env, q_agent, episodes=10000)

    # Greedy Policy 평가 (epsilon=0)
    class GreedyQPolicy:
        def choose_action(self, state):
            return np.argmax(q_agent.Q_table[state])
    greedy_rewards, greedy_replace, greedy_safety = run_policy_full(env, GreedyQPolicy(), quantize=True, episodes=10000)

    plt.figure(figsize=(14, 5))

    # 전체 리워드(왼쪽)
    plt.subplot(1, 2, 1)
    plt.plot(moving_average(fixed_rewards), label="Fixed Policy")
    plt.plot(moving_average(q_rewards), label="Q-Learning")
    plt.plot(moving_average(greedy_rewards), label="Greedy Policy")
    plt.title("Policy Performance Comparison")
    plt.ylabel("Mean Total Reward (Moving Average)")
    plt.legend()

    # 자원 소모량(오른쪽)
    plt.subplot(1, 2, 2)
    plt.plot(moving_average(fixed_replace), label="Fixed Policy")
    plt.plot(moving_average(q_replace), label="Q-Learning")
    plt.plot(moving_average(greedy_replace), label="Greedy Policy")
    plt.title("Resource Usage Comparison")
    plt.ylabel("Water Replacement Count (Moving Average)")
    plt.legend()

    plt.tight_layout()
    plt.show()

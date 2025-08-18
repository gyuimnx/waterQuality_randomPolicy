import numpy as np
import random

def quantize_state(state):
    residualCI, turbidity, ph, remaining_ci, current_step = state
    if ph < 5.8: #pH 낮음
        ph_state = 0
    elif ph > 8.6: #pH 높음
        ph_state = 2
    else: #pH 정상
        ph_state = 1
        
    #탁도 2.8 이하면 0
    turbidity_state = 0 if turbidity <= 2.8 else 1
    
    #잔류염소
    if residualCI < 0.4: #잔류염소 낮음
        residualCI_state = 0
    elif residualCI > 2.0: #잔류염소 높음
        residual_ci_state = 2
    else: #잔류염소 정상
        residualCI_state = 1
    
    #남은 염소(하루 최대 염소 투입 200kg)
    if remaining_ci == 0:
        remaining_ci_state = 0
    elif remaining_ci <= 50:
        remaining_ci_state = 1
    elif remaining_ci <= 100:
        remaining_ci_state = 2
    elif remaining_ci <= 150:
        remaining_ci_state = 3
    else:
        remaining_ci_state = 4
        
    #시간 양자화 -> 아침 오후 저녁으로 다시 해야함
    hour = 9 + (int(current_step) * 10) // 60
    if 9 <= hour < 12:
        time_state = 0
    elif 12 <= hour < 14:
        time_state = 1
    elif 14 <= hour < 17:
        time_state = 2
    else:
        time_state = 3
        
    return (residualCI_state, turbidity_state, ph_state, remaining_ci_state, time_state)

class QAgent:
    def __init__(self, state_shape=(3,2,3,5,4), n_actions=4, alpha=0.1, gamma=0.95, epsilon=0.1, epsilon_decay=0.0, epsilon_min=0.01): #n_action : 0kg, 5kg, 15kg, 25kg
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.Q_table = np.zeros(state_shape + (n_actions,))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min  #나중에 빼도 됨

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q_table[state])

    def learn(self, state, action, reward, next_state):
        best_next = np.max(self.Q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q_table[state + (action,)]
        self.Q_table[state + (action,)] += self.alpha * td_error

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

class FixedIntervalPolicy:
    def choose_action(self, state):
        _, _, _, remaining_ci, current_step = state
        if remaining_ci > 0 and int(current_step) % 3 == 0:
            return 1
        return 0

class RandomPolicy:
    def choose_action(self, state):
        return random.choice([0, 1])

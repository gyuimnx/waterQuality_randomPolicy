import numpy as np
import random

def get_influx_multiplier(hour):
    if 14 <= hour < 17:
        return 1.0
    elif 9 <= hour < 12:
        return 0.7
    elif 12 <= hour < 14 or 17 <= hour < 19:
        return 0.3
    else:
        return 0.0

class WaterParkEnv:
    def __init__(self, max_steps=60, max_ci=200):
        self.max_steps = max_steps
        self.max_ci = max_ci
        #0kg, 5kg, 15kg, 25kg
        self.action_ci = [0, 5, 15, 25]
        self.reset()

    def get_current_guests(self, step):
        hour = 9 + (step * 10) // 60
        if 9 <= hour < 12:
            return int(10000 * 0.3 / 18)
        elif 12 <= hour < 14:
            return int(10000 * 0.1 / 12)
        elif 14 <= hour < 17:
            return int(10000 * 0.5 / 18)
        else:
            return int(10000 * 0.1 / 12)

    def is_all_optimal(self, state): #정상 개수
        residualCI, turbidity, ph, *_ = state
        return (0.4 <= residualCI <= 2.0) and (turbidity <= 2.8) and (5.8 <= ph <= 8.6)

    def is_all_over(self, state): #비정상 개수
        residualCI, turbidity, ph, *_ = state
        return (0.4 < residualCI or residualCI > 2.0) and (turbidity > 2.8) and (ph < 5.8 or ph > 8.6)

    def reset(self):
        self.state = np.array([
            random.uniform(0.4, 2.0), #잔류염소
            random.uniform(0, 2.8), #탁도
            random.uniform(5.8, 8.6), #pH
            self.max_ci, #남은 염소
            0 #스탭
        ])
        self.steps = 0
        self.usedCI_count = 0 #누적 염소 사용량
        self.done = False
        return self.state.copy()

    def step(self, action):
        residualCI, turbidity, ph, remaining_ci, current_step = self.state #잔류염소, 탁도, pH, 남은염소, 현재스탭
        reward = 0
        done = False
        
        #이해 필요
        ci_to_add = self.action_ci[action] #현재 에이전트가 선택한 행동을 실제 투입할 염소 양으로 변환
        
        hour = 9 + (int(current_step) * 10) // 60
        influx_multiplier = get_influx_multiplier(hour)
        
        #---------------------------------------------------------------------여기부터 다시 수정해야함
        
        if action == 1 and remaining_ci > 0:
            self.state = np.array([
                random.uniform(1.0, 1.5),
                random.uniform(0, 2.0),
                random.uniform(6.0, 8.0),
                remaining_ci - 1, #<---------------------------이거 얼마나 사용할지 바꿔야함 근데 잔류염소가 높으면 물 교체 해야함 / 잔류 염소는 염소를 투입할때 10kg 당 0.2mg 정도 증가함. 잔류 염소가 많을땐 염소를 투입하면 안됨
                current_step + 1
            ])
            self.usedCI_count += 1
        elif action == 1 and remaining_ci <= 0:
            self.state[4] += 1
        else:
            #ph +-3.5만큼 변동
            ph_change = random.uniform(-3.0, 3.0) * influx_multiplier
            new_ph = ph + ph_change
            #탁도 최대 +5만큼 변동
            new_turbidity = turbidity + random.uniform(3.0, 5.0) * influx_multiplier
            #잔류염소 최대 +0.1~0.2만큼 변동 <------------------------------------------------ 이거 아님 다시 설계 해야함(아무것도 안해도 증가함)
            new_residualCI = residualCI + random.uniform(0.1, 0.2) * influx_multiplier
            self.state = np.array([
                new_residualCI,
                new_turbidity,
                new_ph,
                remaining_ci,
                current_step + 1
            ])
        self.steps += 1

        # 기준 초과 개수
        exceed_count = 0
        if self.state[0] < 0.4 or self.state[0] > 2.0:
            exceed_count += 1
        if self.state[1] > 2.8:
            exceed_count += 1
        if self.state[2] < 5.8 or self.state[2] > 8.6:
            exceed_count += 1
        
        #자연 변동
        residualCI -= random.uniform(0.05, 0.15) * influx_multiplier #점점 줄어듦
        turbidity += random.uniform(0.5, 1.5) * influx_multiplier #점점 올라감
        ph += random.uniform(-0.2, 0.2) * influx_multiplier #플마
        
        #범위 제한(0 미만 x)
        residualCI = max(0.0, residualCI)
        turbidity = max(0.0, turbidity)
        ph = max(0.0, ph)

        #보상 함수: 기준 초과 개수 및 액션 <------------------------------------이거도 바꿔야함
        if self.is_all_optimal(self.state) and action == 1:
            reward = -1.0  #최적 상태에서 교체(자원 낭비)
        elif exceed_count == 1: #1개 초과
            if action == 1: #교체
                reward = 0.3
            else: #유지
                reward = -0.1
        elif exceed_count == 2: #2개 초과
            if action == 1: #교체
                reward = 0.6
            else: #유지
                reward = -0.7
        elif exceed_count == 3: #3개 모두 초과
            if action == 1: #교체
                reward = 0.8
            else: #유지
                reward = -1.0
        elif self.is_all_optimal(self.state) and action == 0: #모두 정상
            reward = 0.6 #유지

        #교체할 염소가 없는데 교체 시도 <-----------------------------------------이거도
        if action == 1 and remaining_ci <= 0:
            reward -= 0.4

        # 기존 reward 산정 이후 아래 패널티 추가 <-----------------------------------------이거도
        usedCI_penalty = 0 
        if action == 1:  # 물 교체 시도 시
            usedCI_penalty = -0.4 # 교체 한 번 당 페널티
        reward += usedCI_penalty


        if self.state[4] >= self.max_steps or self.steps >= self.max_steps:
            done = True
        self.done = done
        return self.state.copy(), reward, done, {
            'ammonia': self.state[0],
            'turbidity': self.state[1],
            'ph': self.state[2],
            'remaining_ci': self.state[3],
            'step': self.state[4],
            'guests': self.get_current_guests(int(self.state[4]))
        }

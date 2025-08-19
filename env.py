import numpy as np
import random

def get_pollution_factor(hour):
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
        self.max_ci = max_ci #하루 최대 염소 사용량(kg)
        #0kg, 5kg, 15kg, 25kg
        self.action_ci = [0, 5, 20, 30]
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

    # def is_all_optimal(self, state): #정상 개수
    #     residualCI, turbidity, ph, *_ = state
    #     return (0.4 <= residualCI <= 2.0) and (turbidity <= 2.8) and (5.8 <= ph <= 8.6)

    # def is_all_over(self, state): #비정상 개수
    #     residualCI, turbidity, ph, *_ = state
    #     return (0.4 < residualCI or residualCI > 2.0) and (turbidity > 2.8) and (ph < 5.8 or ph > 8.6)

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
        reward = 0.7 #기본 보상(수정필요)
        done = False
        
        #염소 투입
        ci_to_add = self.action_ci[action] #선택한 행동을 실제 염소 사용량으로 변환
        if remaining_ci >= ci_to_add:
            remaining_ci -= ci_to_add
            self.usedCI_count += ci_to_add
            #염소 투입 시 잔류염소 증가(10kg당 0.2mg 가정)
            residualCI += (ci_to_add / 10.0) * 0.2
            turbidity -= ci_to_add * 0.1
            ph -= ci_to_add * 0.1
            
        else:
            #남은 염소 부족 패널티(이거 바꿀 예정)
            reward -= 0.3
        
        #오염 증가(인원 유입)
        hour = 9 + (int(current_step) * 10) // 60
        pollution_factor = get_pollution_factor(hour)

        ph += random.uniform(-0.1, 0.1) * pollution_factor
        turbidity += random.uniform(0.5, 1.0) * pollution_factor
        residualCI -= random.uniform(0.05, 0.1) * pollution_factor

        #자연 복원(환경 회복)
        turbidity -= random.uniform(0.1, 0.3)  #입자 가라앉음
        if ph > 7.0:  #염기성에서 산성
            ph -= random.uniform(0.01, 0.05)
        elif ph < 7.0:  #산성에서 염기성
            ph += random.uniform(0.01, 0.05)

        #음수 방지
        residualCI = max(0.0, residualCI)
        turbidity = max(0.0, turbidity)
        ph = max(0.0, ph)

        #-------------보상 계산-------------
        #잔류염소 패널티
        if residualCI > 2.0:
            reward -= (residualCI - 2.0)
        elif residualCI < 0.4:
            reward -= (0.4 - residualCI)
        else:
            reward += 0

        #탁도 패널티
        if turbidity > 2.8:
            reward -= (turbidity - 2.8)
        else:
            reward += 0

        #pH 패널티
        if ph > 8.6:
            reward -= (ph - 8.6)
        elif ph < 5.8:
            reward -= (5.8 - ph)
        else:
            reward += 0

        #자원 초과 사용 패널티
        if self.usedCI_count > self.max_ci:
            excess = self.usedCI_count - self.max_ci
            reward -= excess * 0.1

        #-------------상태 업데이트-------------
        current_step += 1
        self.state = np.array([
            residualCI,
            turbidity,
            ph,
            remaining_ci,
            current_step
        ])
        self.steps += 1

        #종료 조건
        if current_step >= self.max_steps or self.steps >= self.max_steps: #하루가 끝났으면 에피소드 종료
            done = True
        self.done = done

        return self.state.copy(), reward, done, {
            'residualCI': residualCI,
            'turbidity': turbidity,
            'ph': ph,
            'remaining_ci': remaining_ci,
            'step': current_step,
            'used_ci': self.usedCI_count,
            'guests': self.get_current_guests(int(current_step))
        }
        # if action == 1 and remaining_ci > 0:
        #     self.state = np.array([
        #         random.uniform(1.0, 1.5),
        #         random.uniform(0, 2.0),
        #         random.uniform(6.0, 8.0),
        #         remaining_ci - 1, #<---------------------------이거 얼마나 사용할지 바꿔야함 근데 잔류염소가 높으면 물 교체 해야함 / 잔류 염소는 염소를 투입할때 10kg 당 0.2mg 정도 증가함. 잔류 염소가 많을땐 염소를 투입하면 안됨
        #         current_step + 1
        #     ])
        #     self.usedCI_count += 1
        # elif action == 1 and remaining_ci <= 0:
        #     self.state[4] += 1
        # else:
        #     #ph +-3.5만큼 변동
        #     ph_change = random.uniform(-3.0, 3.0) * influx_multiplier
        #     new_ph = ph + ph_change
        #     #탁도 최대 +5만큼 변동
        #     new_turbidity = turbidity + random.uniform(3.0, 5.0) * influx_multiplier
        #     #잔류염소 최대 +0.1~0.2만큼 변동 <------------------------------------------------ 이거 아님 다시 설계 해야함(아무것도 안해도 증가함)
        #     new_residualCI = residualCI + random.uniform(0.1, 0.2) * influx_multiplier
        #     self.state = np.array([
        #         new_residualCI,
        #         new_turbidity,
        #         new_ph,
        #         remaining_ci,
        #         current_step + 1
        #     ])
        # self.steps += 1

        # # 기준 초과 개수
        # exceed_count = 0
        # if self.state[0] < 0.4 or self.state[0] > 2.0:
        #     exceed_count += 1
        # if self.state[1] > 2.8:
        #     exceed_count += 1
        # if self.state[2] < 5.8 or self.state[2] > 8.6:
        #     exceed_count += 1
        
        # #자연 변동
        # residualCI -= random.uniform(0.05, 0.15) * influx_multiplier #점점 줄어듦
        # turbidity += random.uniform(0.5, 1.5) * influx_multiplier #점점 올라감
        # ph += random.uniform(-0.2, 0.2) * influx_multiplier #플마
        
        # #범위 제한(0 미만 x)
        # residualCI = max(0.0, residualCI)
        # turbidity = max(0.0, turbidity)
        # ph = max(0.0, ph)

        # #보상 함수: 기준 초과 개수 및 액션 <------------------------------------이거도 바꿔야함
        # if self.is_all_optimal(self.state) and action == 1:
        #     reward = -1.0  #최적 상태에서 교체(자원 낭비)
        # elif exceed_count == 1: #1개 초과
        #     if action == 1: #교체
        #         reward = 0.3
        #     else: #유지
        #         reward = -0.1
        # elif exceed_count == 2: #2개 초과
        #     if action == 1: #교체
        #         reward = 0.6
        #     else: #유지
        #         reward = -0.7
        # elif exceed_count == 3: #3개 모두 초과
        #     if action == 1: #교체
        #         reward = 0.8
        #     else: #유지
        #         reward = -1.0
        # elif self.is_all_optimal(self.state) and action == 0: #모두 정상
        #     reward = 0.6 #유지

        # #교체할 염소가 없는데 교체 시도 <-----------------------------------------이거도()
        # if action == 1 and remaining_ci <= 0:
        #     reward -= 0.4

        # # 기존 reward 산정 이후 아래 패널티 추가 <-----------------------------------------이거도
        # usedCI_penalty = 0 
        # if action == 1:  # 물 교체 시도 시
        #     usedCI_penalty = -0.4 # 교체 한 번 당 페널티
        # reward += usedCI_penalty


        # if self.state[4] >= self.max_steps or self.steps >= self.max_steps:
        #     done = True
        # self.done = done
        # return self.state.copy(), reward, done, {
        #     'ammonia': self.state[0],
        #     'turbidity': self.state[1],
        #     'ph': self.state[2],
        #     'remaining_ci': self.state[3],
        #     'step': self.state[4],
        #     'guests': self.get_current_guests(int(self.state[4]))
        # }

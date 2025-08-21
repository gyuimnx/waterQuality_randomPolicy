#env.py
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
        reward = 0.0 #기본 보상(수정필요)---------------------------------------------------------------------------------------기본 보상 수정 필요
        done = False
        
        #염소 투입
        ci_to_add = self.action_ci[action] #선택한 행동을 실제 염소 사용량으로 변환
        if remaining_ci >= ci_to_add:
            remaining_ci -= ci_to_add
            self.usedCI_count += ci_to_add
            #염소 투입 시 잔류염소 증가(10kg당 0.2mg 가정)
            residualCI += (ci_to_add / 10.0) * 0.2 
            # # 자원 소모 패널티: 투입한 염소량(kg)에 비례해 보상 차감-------------------------------------------------------여기서
            # w_ci_use = 0.03   # kg당 -0.03 정도 (예: 20kg 넣으면 -0.6)
            reward -= 0.1 * ci_to_add#-------------------------------------------------------이까지 수정함(지워도됨)
            
            turbidity -= ci_to_add * 0.1
            ph -= ci_to_add * 0.05
        else:
            #남은 염소 부족 패널티(이거 바꿀 예정)
            reward -= 0.2
        
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
        # 정상 상태 보상 (잔류염소, 탁도, pH 모두 정상 범위일 때)------------------------------------------------
        if 0.4 <= residualCI <= 2.0 and turbidity <= 2.8 and 5.8 <= ph <= 8.6:
            reward += 1.0
        else:
            reward -= 0.5
            
        #잔류염소 패널티
        if residualCI > 2.0:
            reward -= (residualCI - 2.0)
        elif residualCI < 0.4:
            reward -= (0.4 - residualCI)
        else:
            reward += 0.0

        #탁도 패널티
        if turbidity > 2.8:
            reward -= (turbidity - 2.8)
        else:
            reward += 0.0

        #pH 패널티
        if ph > 8.6:
            reward -= (ph - 8.6)
        elif ph < 5.8:
            reward -= (5.8 - ph)
        else:
            reward += 0.0

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

        # #디버깅
        # print(f"[STEP {current_step}] Action={action}, Reward={reward:.3f}, CI={residualCI:.2f}, Turb={turbidity:.2f}, pH={ph:.2f}")

        return self.state.copy(), reward, done, {
            'residualCI': residualCI,
            'turbidity': turbidity,
            'ph': ph,
            'remaining_ci': remaining_ci,
            'step': current_step,
            'used_ci': self.usedCI_count,
            'guests': self.get_current_guests(int(current_step))
        }
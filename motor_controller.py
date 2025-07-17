# Copyright (c) 2024 Sungkyunkwan University AutomationLab
#
# Authors:
# - Gyuhyeon Hwang <rbgus7080@g.skku.edu>, Hobin Oh <hobin0676@daum.net>, Minkwan Choi <arbong97@naver.com>, Hyeonjin Sim <nufxwms@naver.com>
# - url: https://micro.skku.ac.kr/micro/index.do

import time
from threading import Lock
import spidev
import keyboard
import numpy as np

class MotorController:
    def __init__(self, motors):
        # 기본 모터 설정
        self.motors = motors
        self.size = 600600  # 2ms
        self._left_speed = 0
        self._right_speed = 0
        self._steering_speed = 0
        self.steering_angle = 0
        
        self.current_duty = self.size // 2  # 현재 duty 값 (50%)
        self.min_duty = self.size // 2      # 최소 duty 값 (50%)
        self.max_duty = int(self.size * 0.8)  # 최대 duty 값 (80%)
        self.duty_step = int(self.size * 0.02)  # duty 증가량 (2%)
        self.last_steering_time = time.time()
        
        
        
                # 조향 딜레이 관련 변수 추가
        #self.angle_change_delay = 0.5  # 각도 변화 시 딜레이 (초)
        #self.last_angle_change_time = time.time()
        #self.last_target_angle = 0
        
        
        
        
        
        
        # 제어 변수
        # self.auto_duty = self.min_duty
        self.manual_steering_angle = 0
        self.manual_speed = 0
        
        # SPI 설정
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = 20000000
        self.spi.mode = 0b00
        
        # 저항 값 범위 설정
        self.resistance_most_left = 3400
        self.resistance_most_right = 2800
    @property
    def steering_speed(self):
        return self._steering_speed

    @steering_speed.setter
    def steering_speed(self, value):
        self._steering_speed = value
        self.control_motors(self._steering_speed)  # 속도 변경 시 자동으로 반영
        
    @property
    def left_speed(self):
        return self._left_speed

    @left_speed.setter
    def left_speed(self, value):
        self._left_speed = value
        self.set_left_speed(self._left_speed)  # 속도 변경 시 자동으로 반영

    @property
    def right_speed(self):
        return self._right_speed

    @right_speed.setter
    def right_speed(self, value):
        self._right_speed = value
        self.set_right_speed(self._right_speed)  # 속도 변경 시 자동으로 반영

    def init_motors(self):
        """모터 초기화"""
        for name, motor in self.motors.items():
            motor.write(0x00, self.size)     # size
            motor.write(0x04, self.min_duty)  # 초기 duty 50%
            motor.write(0x08, 0)             # valid

    def reset_motor_values(self):
        """모터 값 안전 초기화"""
        self.left_speed = 0
        self.right_speed = 0
        self.steering_speed = 0
        self.steering_angle = 0
        self.manual_speed = 0
        self.manual_steering_angle = 0
        self.current_duty = self.min_duty
        
        # 모든 모터 정지
        for motor in self.motors.values():
            motor.write(0x08, 0)
        
        # duty 값 초기화
        for motor in self.motors.values():
            motor.write(0x04, self.min_duty)

    def right(self, steering_speed, control_mode=1):
        """우회전 제어"""
        if control_mode == 1:  # 자율주행 모드
            duty_percent = abs(steering_speed) / 1
            duty = int(self.size * duty_percent)            
            # duty = self.auto_duty
        elif control_mode == 2:  # 수동 주행 모드
            current_time = time.time()
            if current_time - self.last_steering_time > 0.05:
                self.current_duty = min(self.max_duty, self.current_duty + self.duty_step)
                self.last_steering_time = current_time
            duty = self.current_duty
        else:
            duty_percent = abs(steering_speed) / 5
            duty = int(self.size * duty_percent)
            
        print("right duty", duty)    
        self.motors['motor_4'].write(0x08, 0)  # valid  steering_left
        self.motors['motor_5'].write(0x08, 1)  # valid  steering_right
        self.motors['motor_5'].write(0x04, duty)

    def left(self, steering_speed, control_mode=1):
        """좌회전 제어"""
        if control_mode == 1:  # 자율주행 모드
            duty_percent = abs(steering_speed) / 20
            duty = int(self.size * duty_percent)
        elif control_mode == 2:  # 수동 주행 모드
            current_time = time.time()
            if current_time - self.last_steering_time > 0.05:
                self.current_duty = min(self.max_duty, self.current_duty + self.duty_step)
                self.last_steering_time = current_time
            duty = self.current_duty
        else: 
            duty_percent = abs(steering_speed) / 3
            duty = int(self.size * duty_percent)
            
        print("left duty", duty)    
        self.motors['motor_5'].write(0x08, 0)  # valid  steering_right
        self.motors['motor_4'].write(0x08, 1)  # valid  steering_left
        self.motors['motor_4'].write(0x04, duty)

    def stay(self, steering_speed, control_mode=1):
        """중립 상태 유지"""
        if control_mode == 1:  # 자율주행 모드
            duty_percent = abs(steering_speed) / 5
            duty = int(self.size * duty_percent)
        elif control_mode == 2:  # 수동 주행 모드
            self.current_duty = self.min_duty
            duty = self.current_duty
        else:
            duty_percent = abs(steering_speed) / 5
            duty = int(self.size * duty_percent)
            
        print("stay duty", duty)    
        self.motors['motor_5'].write(0x08, 0)  # valid  steering_right
        self.motors['motor_4'].write(0x08, 0)  # valid  steering_left
        self.motors['motor_5'].write(0x04, duty)
        self.motors['motor_4'].write(0x04, duty)
# motor_controller.py
  
    def center(self, speed=0, control_mode=1, settle_time=1.0):
        """
        Bring steering to 0° using ADC‑feedback.
        • speed: PWM duty 기준이 되는 속도값
        • settle_time: 보정에 허용할 최대 시간(sec)
        """
        self.steering_speed = speed
        self.steering_angle = 0          # 목표각 0°
        start = time.time()

        while time.time() - start < settle_time:
            self.control_motors(angle=0, control_mode=control_mode)
            time.sleep(0.02)             # 20 ms 주기

    def set_left_speed(self, speed):
        """왼쪽 모터 속도 설정"""
        duty_percent = abs(speed) / 85
        duty = int(self.size * duty_percent)
        print("set_left_speed duty", duty)
        
        self.motors['motor_0'].write(0x04, duty)
        self.motors['motor_1'].write(0x04, duty)
        
        if speed > 0:
            self.motors['motor_0'].write(0x08, 0)
            self.motors['motor_1'].write(0x08, 1)
        else:
            self.motors['motor_0'].write(0x08, 1)
            self.motors['motor_1'].write(0x08, 0)

    def set_right_speed(self, speed):
        """오른쪽 모터 속도 설정"""
        duty_percent = abs(speed) / 85
        duty = int(self.size * duty_percent)
        print("set_right_speed duty", duty)

        self.motors['motor_3'].write(0x04, duty)
        self.motors['motor_2'].write(0x04, duty)
        
        if speed > 0:
            self.motors['motor_3'].write(0x08, 0)
            self.motors['motor_2'].write(0x08, 1)
        else:
            self.motors['motor_3'].write(0x08, 1)
            self.motors['motor_2'].write(0x08, 0)

    def read_adc(self):
        """ADC 값 읽기"""
        adc_response = self.spi.xfer2([0x00, 0x00])
        adc_value = ((adc_response[0] & 0x0F) << 8) | adc_response[1]
        return adc_value 

    def map_value(self, x, in_min, in_max, out_min, out_max):
        """
        x를 in_min~in_max 범위에서 out_min~out_max 범위로 매핑
        """
        if x <= in_min:
            return out_max
        elif x >= in_max:
            return out_min
        else:
            # in_min과 in_max 사이일 경우, x가 커질수록 결과가 선형적으로 감소하도록 계산
            return (in_max - x) * (out_max - out_min) / (in_max - in_min) + out_min

    def map_angle_to_range(self, angle):
        """각도를 모터 제어 범위로 매핑"""
        abs_angle = angle
        if (-1 <= abs_angle <= 1) :
            return 0
        elif (-6 < abs_angle < -1) :
            return -2
        elif abs_angle <= -6 :
            return -4
        elif abs_angle >= 1 :
            return 20



    def control_motors(self, angle=None, control_mode=1):
        
        
        #current_time = time.time()

        
        
        
        
        
        
        """모터 전체 제어"""
        mapped_resistance = self.map_value(
            self.read_adc(),
            self.resistance_most_right,
            self.resistance_most_left,
            -20, 20
        )
        
        if angle is not None:
            target_angle = self.map_angle_to_range(angle)
        else:
            target_angle = self.steering_angle
            
            
            # 각도 변화 감지 및 딜레이 적용
            #if abs(target_angle - self.last_target_angle) > 0.1:  # 각도 변화가 있을 때
                #if current_time - self.last_angle_change_time < self.angle_change_delay:
                   # return  # 딜레이 중이면 제어하지 않음
                #self.last_angle_change_time = current_time
                #self.last_target_angle = target_angle
            
            
            
        tolerance = 0.1
        if abs(mapped_resistance - target_angle) <= tolerance:
            self.stay(self.steering_speed, control_mode)
        elif mapped_resistance > target_angle:
            self.left(self.steering_speed, control_mode)
        else:
            self.right(self.steering_speed, control_mode)

    def handle_manual_control(self):
        """수동 주행 모드에서의 키보드 입력 처리"""
        if keyboard.is_pressed('w'):
            self.left_speed = min(self.left_speed + 1, 100)
            self.right_speed = min(self.right_speed + 1, 100)
            
        if keyboard.is_pressed('s'):
            self.left_speed = max(self.left_speed - 1, -100)
            self.right_speed = max(self.right_speed - 1, -100)
            
        if keyboard.is_pressed('a'):
            self.steering_angle = min(self.steering_angle - 1, 20)
            
        if keyboard.is_pressed('d'):
            self.steering_angle = max(self.steering_angle + 1, -20)
            
        if keyboard.is_pressed('r'):
            self.left_speed = 0
            self.right_speed = 0
            self.steering_angle = 0

        # 모터 제어 적용
        self.set_left_speed(self.left_speed)
        self.set_right_speed(self.right_speed)
        self.control_motors(control_mode=2)

#     def control_motors_parking(self, angle, speed, direction='straight'):
#         # _turn_left, _turn_right, _straight_steering, _set_steering_angle
#         # left, right, straight, steering

#         """주차 모드에서의 모터 제어"""
#         if direction == 'left':
#             self.left_speed = speed
#             self.left(speed, 3)
#         elif direction == 'right':
#             self.right_speed = speed
#             self.right(speed, 3)
#         elif direction == 'straight_left':
#             self.left_speed = speed
#             self.right_speed = speed
#             self.steering_angle = angle
#             self.right(speed, 3)
#         elif direction == 'straight_right':
#             self.left_speed = speed
#             self.right_speed = speed
#             self.steering_angle = angle
#             self.left(speed, 3)
#         elif direction == 'center':
#             # self.left_speed = speed
#             # self.right_speed = speed
#             self.center(speed,3)   # ★
#             return
#             # 주차용 정렬만 하면 달리지는 않으므로
#             # 필요하면 이후 _move_forward/_backward 호출


#         self.set_left_speed(self.left_speed)
#         self.set_right_speed(self.right_speed)
    def control_motors_parking(
            self,
            angle               = 0,
            speed               = 0,
            direction           = 'center',
            settle_time         = 1.0,
            coarse              = False  # 피드백용 정밀/코스 설정
        ):
        """
        Parking‑mode motor control
        • angle       : 원하는 조향 각도 (degree)
        • speed       : 조향용 PWM 기준 duty (0~100)
        • direction   : 'left' | 'right' | 'straight_left' | 'straight_right'
                        'center' : 핸들 0° 로 수렴 (피드백 루프)
        • settle_time : 'center' 모드 정렬 허용 시간(sec)
        • coarse      : True  => 자율주행 coarse map
                        False => 정밀 매핑 (주차/수동 권장)
        """
        if direction == 'left':
            self.left_speed  = speed
            self.left(speed, 3)                     # 조향만
            self.set_left_speed(self.left_speed)    # 구동 PWM
        elif direction == 'right':
            self.right_speed = speed
            self.right(speed, 3)
            self.set_right_speed(self.right_speed)

        elif direction == 'straight_left':
            # 핸들을 angle 만큼 틀고 직진
            self.steering_angle = angle
            self.right(speed, 3)                    # 오른쪽 휠로 핸들 돌림
            self.left_speed = self.right_speed = speed
            self.set_left_speed(speed)
            self.set_right_speed(speed)

        elif direction == 'straight_right':
            self.steering_angle = angle
            self.left(speed, 3)
            self.left_speed = self.right_speed = speed
            self.set_left_speed(speed)
            self.set_right_speed(speed)

        elif direction == 'center':
            # 핸들을 0°로 ±tolerance 안에 들어올 때까지 반복
            tolerance = 0.3          # ±0.3° 안이면 멈춤 (원하면 조정)
            settle_time = 1.0        # 최대 1초 동안만 보정
            steer_pwm   = speed      # 예: 30 정도면 충분

            start = time.time()
            while time.time() - start < settle_time:
                adc_deg = self.map_value(               # 현재 핸들 각도(‑20~20°)
                    self.read_adc(),
                    self.resistance_most_right,
                    self.resistance_most_left,
                    -20, 20
                )

                if abs(adc_deg) <= tolerance:
                    # 이미 중앙이면 stay()로 valid 비트 OFF
                    self.stay(steer_pwm, 3)
                    break

                if adc_deg > 0:
                    # 핸들이 오른쪽으로 치우쳤으니 왼쪽으로 돌려야 함
                    self.left(steer_pwm, 3)     # 모터 4에 PWM 인가
                else:
                    # 핸들이 왼쪽으로 치우침 → 오른쪽으로 돌려야
                    self.right(steer_pwm, 3)    # 모터 5에 PWM 인가

                time.sleep(0.02)                # 20 ms 주기

            # 핸들만 맞췄으므로 구동 바퀴는 건드리지 않는다
            return


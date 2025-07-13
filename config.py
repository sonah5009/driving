# Copyright (c) 2024 Sungkyunkwan University AutomationLab
#
# Authors:
# - Gyuhyeon Hwang <rbgus7080@g.skku.edu>, Hobin Oh <hobin0676@daum.net>, Minkwan Choi <arbong97@naver.com>, Hyeonjin Sim <nufxwms@naver.com>
# - url: https://micro.skku.ac.kr/micro/index.do

import numpy as np

MOTOR_ADDRESSES = {
    'motor_0': 0x00A0000000,
    'motor_1': 0x00A0010000,
    'motor_2': 0x00A0020000,
    'motor_3': 0x00A0030000,
    'motor_4': 0x00A0040000,
    'motor_5': 0x00A0050000
}

ADDRESS_RANGE = 0x10000

# 초음파 센서 주소 설정 (주차 시스템용)
ULTRASONIC_ADDRESSES = {
    'ultrasonic_0': 0x00B0000000,  # 전방 우측
    'ultrasonic_1': 0x00B0010000,  # 중간 좌측
    'ultrasonic_2': 0x00B0020000,  # 중간 우측
    'ultrasonic_3': 0x00B0030000,  # 후방 좌측
    'ultrasonic_4': 0x00B0040000   # 후방 우측
}


# 초음파 센서 레지스터 맵
ULTRASONIC_REGISTERS = {
    'DISTANCE_DATA': 0x00    # 거리 데이터 (4바이트, mm 단위)
}

# 초음파 센서 설정
ULTRASONIC_CONFIG = {
    'MEASUREMENT_INTERVAL': 0.1,  # 측정 간격 (초)
    'TIMEOUT_THRESHOLD': 5000,    # 타임아웃 임계값 (ms)
    'MIN_DISTANCE': 20,           # 최소 측정 거리 (mm)
    'MAX_DISTANCE': 4000,         # 최대 측정 거리 (mm)
    'SOUND_SPEED': 343000         # 음속 (mm/s)
}

# YOLO configurations
anchor_list = [10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]
anchors = np.array(anchor_list).reshape(-1, 2)
classes_path = "../xmodel/lane_class.txt"


# Kanayama 제어기 파라미터
KANAYAMA_CONFIG = {
    'K_y': 0.3,        # 횡방향 오차 게인 0.3
    'K_phi': 0.9,      # 방향각 오차 게인 0.9
    'L': 0.55,          # 휠베이스 (m)
    'lane_width': 3.5, # 차선 폭 (m)
    'v_r': 20.0        # 기준 속도 (30.0에서 20.0으로 낮춤)
}

# 히스토리 관리 설정
HISTORY_CONFIG = {
    'max_history_size': 10,      # 최대 히스토리 크기
    'avg_window_size': 5,        # 평균 계산에 사용할 프레임 수
    'max_no_lane_frames': 5,     # 최대 차선 미검출 프레임 수
    'default_steering_angle': 0.0,  # 기본 조향각 (직진)
    'smoothing_factor': 0.2      # 스무딩 팩터 (0~1, 높을수록 부드러움)
}



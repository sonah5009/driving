# Copyright (c) 2024 Sungkyunkwan University AutomationLab
#
# Authors:
# - Gyuhyeon Hwang <rbgus7080@g.skku.edu>, Hobin Oh <hobin0676@daum.net>, Minkwan Choi <arbong97@naver.com>, Hyeonjin Sim <nufxwms@naver.com>
# - url: https://micro.skku.ac.kr/micro/index.do

from pynq import Overlay, MMIO, PL, allocate
from pynq.lib.video import *
from pynq_dpu import DpuOverlay
import cv2
import time
import spidev
import keyboard
from driving_system_controller import DrivingSystemController
from parking_system_controller import ParkingSystemController
from image_processor import ImageProcessor
from config import MOTOR_ADDRESSES, ULTRASONIC_ADDRESSES, ADDRESS_RANGE
from AutoLab_lib import init
import threading
from hard_code import HardCodeController


init()
# Initialize SPI
spi0 = spidev.SpiDev()
spi0.open(0, 0)
spi0.max_speed_hz = 20000000
spi0.mode = 0b00

# 자율주행 모드 뒷바퀴 & 조향 속도 설정 (0 ~ 100)
speed = 5  # 50에서 30으로 낮춤 (더 안전한 속도)
steering_speed = 50
motors = {}
parking_mode = False

# 주행 시스템 모터 값
for name, addr in MOTOR_ADDRESSES.items():
    motors[name] = MMIO(addr, ADDRESS_RANGE)

# 주차 시스템 초음파 센서 값
ultrasonic_sensors = {}
for name, addr in ULTRASONIC_ADDRESSES.items():
    ultrasonic_sensors[name] = MMIO(addr, ADDRESS_RANGE)

def load_dpu():
    global dpu, input_data, output_data, shapeIn, shapeOut0, shapeOut1
    
    overlay = DpuOverlay("../dpu/dpu.bit")
    overlay.load_model("../xmodel/tiny-yolov3_coco_256.xmodel")
    
    dpu = overlay.runner
    
    return overlay, dpu

def main():
    overlay = load_dpu()
    controller = DrivingSystemController(overlay, dpu, motors, speed, steering_speed, parking_mode)
    # 기존 ParkingSystemController 대신 HardCodeController 사용
    # parking_controller = ParkingSystemController(controller.motor_controller, ultrasonic_sensors)
    parking_config = {
        'forward_speed': 45.0,
        'backward_speed': 45.0,
        'steering_speed': 50.0,
        'left_turn_angle': -50,
        'right_turn_angle': 50
    }
    parking_controller = HardCodeController(controller.motor_controller, parking_config)
    
    # HardCodeController만 사용할 것이므로 주차 관련 변수/스레드 제거
    # parking_thread = None
    # monitor_thread = None
    # threads_started = False
    
    try:
        # 카메라 초기화
        camera_index = 0

        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("카메라를 열 수 없습니다.")
            return

        # 시작 시 모드 선택
        print("\n주행 모드를 선택하세요:")
        print("1: 자율주행 모드")
        print("2: 수동주행 모드")
        print("P: 주차 모드 진입")
        
        while True:
            # 키보드 입력 처리
            if keyboard.is_pressed('1'):
                controller.switch_mode(1)
                break
            elif keyboard.is_pressed('2'):
                controller.switch_mode(2)
                break
            elif keyboard.is_pressed('p'):
                print("커스텀 시퀀스 준비됨 (Space 키를 눌러 실행)")
                break
            time.sleep(0.1)

        # 제어 안내 출력
        print("\n키보드 제어 안내:")
        print("Space: 주행 및 주차 시작")
        print("1/2: 자율주행/수동주행 모드 전환")
        if controller.control_mode == 2:    
            print("\t수동 주행 제어:")
            print("\tW/S: 전진/후진")
            print("\tA/D: 좌회전/우회전")
            print("\tR: 긴급 정지")
            
        print("Q: 프로그램 종료\n")


        # 센서 데이터 출력 시간 제어용 변수
        last_sensor_print_time = time.time()
        

        while True:
            # space 키를 누르면 커스텀 시퀀스 실행 후 break
            if keyboard.is_pressed('space'):
                time.sleep(0.3)  # 디바운싱
                parking_controller.run_custom_sequence()
                print("🔄 커스텀 주차 시퀀스 시작됨")
                break
            
            if keyboard.is_pressed('q'):
                print("\n프로그램을 종료합니다.")
                break

            # 프레임 처리 (주차 시스템이 비활성화된 경우에만)
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break
            controller.process_and_control(frame)
            

    except KeyboardInterrupt:
        print("\n사용자에 의해 중지되었습니다.")
    finally:
        # 리소스 정리
        cap.release()
        cv2.destroyAllWindows()
        controller.stop_driving()
        print("모든 시스템이 정리되었습니다.")

if __name__ == "__main__":
    main()
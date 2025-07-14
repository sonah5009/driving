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



init()
# Initialize SPI
spi0 = spidev.SpiDev()
spi0.open(0, 0)
spi0.max_speed_hz = 20000000
spi0.mode = 0b00

# 자율주행 모드 뒷바퀴 & 조향 속도 설정 (0 ~ 100)
speed = 30  # 50에서 30으로 낮춤 (더 안전한 속도)
steering_speed = 50
motors = {}

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
    controller = DrivingSystemController(overlay, dpu, motors, speed, steering_speed)
    parking_controller = ParkingSystemController(controller.motor_controller, ultrasonic_sensors)
    
    # 스레드 관리 변수 추가
    parking_thread = None
    monitor_thread = None
    threads_started = False
    
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
            if keyboard.is_pressed('1'):
                controller.switch_mode(1)
                break
            elif keyboard.is_pressed('2'):
                controller.switch_mode(2)
                break
            elif keyboard.is_pressed('p'):
                parking_controller.enter_parking_mode()
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
            
        print("P: 주차 강제 종료")
        print("Q: 프로그램 종료\n")


        # 센서 데이터 출력 시간 제어용 변수
        last_sensor_print_time = time.time()

        while True:
            # 키보드 입력 처리
            if keyboard.is_pressed('space'):
                time.sleep(0.3)  # 디바운싱
                if parking_controller.is_parking_active:
                    print("주차 시작 시, 중도 정지는 안되며")
                    print("p를 눌러 주차 모드를 종료하세요")
                elif parking_controller.is_parking_mode:
                    parking_controller.start_parking()
                    # print("🚗 주차 시작!")
                elif controller.is_running:
                    controller.stop_driving()
                    # print("주행 중지됨")
                elif not controller.is_running:
                    controller.start_driving()
                    # print("주행 시작됨")
            
            elif keyboard.is_pressed('1') or keyboard.is_pressed('2'):
                if parking_controller.is_parking_active or parking_controller.is_parking_mode:
                    print("🚗 주차 모드가 활성화되어 있습니다.")
                    print("   → P 키를 눌러 주차 모드를 먼저 종료하세요.")
                else:
                    prev_mode = controller.control_mode
                    new_mode = 1 if keyboard.is_pressed('1') else 2
                    if prev_mode != new_mode:
                        controller.switch_mode(new_mode)
                        if new_mode == 2:
                            print("\n수동 주행 제어:")
                            print("W/S: 전진/후진")
                            print("A/D: 좌회전/우회전")
                            print("R: 긴급 정지")
                time.sleep(0.3)  # 디바운싱
            
            elif keyboard.is_pressed('p'):
                time.sleep(0.3)  # 디바운싱
                if controller.is_running:
                    print("🚗 자율주행이 실행 중입니다.")
                    print("   → Space 키를 눌러 주행을 먼저 중지하세요.")
                elif parking_controller.is_parking_mode:
                    parking_controller.exit_parking_mode()
                    print("🛑 주차가 강제 종료되었습니다.")
                    # 스레드 상태 초기화
                    threads_started = False
                    parking_thread = None
                    monitor_thread = None
                    print("\n주행 모드를 다시 선택하세요:")
                    print("1: 자율주행 모드")
                    print("2: 수동주행 모드")
                    print("P: 주차 모드 진입")
                    continue
                    
                else:
                    parking_controller.enter_parking_mode()
                    print("🚗 주차 모드 진입")
                    print("   → Space 키를 눌러 주차를 시작하세요.")
            
            if keyboard.is_pressed('q'):
                print("\n프로그램을 종료합니다.")
                break

            # 주차 시스템 실행 (활성화된 경우)
            if parking_controller.is_parking_active and not threads_started:
                # 스레드가 아직 시작되지 않은 경우에만 새로 생성
                parking_thread = threading.Thread(target=parking_controller.parking_cycle_thread, daemon=True)
                monitor_thread = threading.Thread(target=parking_controller.status_monitor_thread, daemon=True)

                parking_thread.start()
                monitor_thread.start()
                threads_started = True
                
                print("🔄 주차 스레드 시작됨")
                
                # 주차 시스템이 활성화된 경우 자율주행 처리를 건너뜀
                continue

            # 주차 시스템이 아예 종료됐을 경우(주차가 끝났거나, 주차모드 종료) 스레드 상태 초기화
            elif not (parking_controller.is_parking_active or parking_controller.is_parking_mode) and threads_started:
                threads_started = False
                parking_thread = None
                monitor_thread = None
                print("🔄 주차 스레드 중지됨")

                # 주차모드에서 주차가 중간에 중지되거나 종료된 거라 카메라 안씀
                # 그래서 뒤에 있는 코드 실행하지 않고 넘어가기 위해 continue 사용해야하는 지 고민 중
                continue


            # 프레임 처리 (주차 시스템이 비활성화된 경우에만)
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break

            # 이미지 처리 및 차량 제어 (주차 시스템이 비활성화된 경우에만)
            controller.process_and_control(frame)
            

    except KeyboardInterrupt:
        print("\n사용자에 의해 중지되었습니다.")
    finally:
        # 리소스 정리
        cap.release()
        cv2.destroyAllWindows()
        controller.stop_driving()
        parking_controller.stop_parking()
        
        # 스레드 정리
        if parking_thread and parking_thread.is_alive():
            parking_controller.stop_parking()  # 스레드 종료 신호
            parking_thread.join(timeout=2.0)  # 2초 대기
        if monitor_thread and monitor_thread.is_alive():
            monitor_thread.join(timeout=2.0)  # 2초 대기
            
        print("모든 시스템이 정리되었습니다.")

if __name__ == "__main__":
    main()
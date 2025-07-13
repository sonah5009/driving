# Copyright (c) 2024 Sungkyunkwan University AutomationLab
#
# Authors:
# - Gyuhyeon Hwang <rbgus7080@g.skku.edu>, Hobin Oh <hobin0676@daum.net>, Minkwan Choi <arbong97@naver.com>, Hyeonjin Sim <nufxwms@naver.com>
# - url: https://micro.skku.ac.kr/micro/index.do

from pynq import Overlay, MMIO, PL, allocate
from pynq.lib.video import *
from pynq_dpu import DpuOverlay
import cv2
import numpy as np
import time
import os
import spidev
import keyboard
import matplotlib.pyplot as plt
from driving_system_controller import DrivingSystemController
from parking_system_controller import ParkingSystemController
from image_processor import ImageProcessor
from config import MOTOR_ADDRESSES, ULTRASONIC_ADDRESSES, ADDRESS_RANGE
from AutoLab_lib import init

# Jupyter 환경 감지
def is_jupyter_environment():
    """Jupyter 환경인지 확인"""
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook
            return True
        elif shell == 'TerminalInteractiveShell':  # IPython terminal
            return False
        else:
            return False
    except NameError:
        return False

# matplotlib 설정 (Jupyter 환경에서만)
if is_jupyter_environment():
    plt.ion()  # 인터랙티브 모드 활성화
    print("Jupyter 환경에서 실행 중입니다. matplotlib을 사용하여 이미지를 표시합니다.")

def draw_boxes_on_bev(bev_img, bev_boxes, color=(0, 255, 0)):
    """BEV 영상 위에 바운딩 박스를 그리는 함수"""
    if bev_img is None:
        return None
    
    img = bev_img.copy()
    for box in bev_boxes:
        x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
        class_name = box['class']
        score = box['score']
        
        # 박스 그리기
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # 클래스명과 점수 표시
        label = f"{class_name}: {score:.2f}"
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img

def visualize_results(processed_info, frame_count=0):
    """결과를 시각화하는 함수"""
    processed_image = processed_info['processed_image']
    bev_image = processed_info['bev_image']
    bev_boxes = processed_info['bev_boxes']
    lane_info = processed_info['lane_info']
    steering_angle = processed_info['steering_angle']
    calculated_speed = processed_info['speed']
    processing_time = processed_info['processing_time']
    
    if is_jupyter_environment():
        # Jupyter 환경에서 matplotlib 사용
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Autonomous Driving Visualization - Frame {frame_count}', fontsize=16)
        
        # 1. 원본 처리된 이미지
        #axes[0, 0].imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f'Processed Image\nSteering: {steering_angle:.1f}°, Speed: {calculated_speed:.1f} m/s')
        axes[0, 0].axis('off')
        
        # 2. BEV 영상 (바운딩 박스 없음)
        if bev_image is not None:
            #axes[0, 1].imshow(cv2.cvtColor(bev_image, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title('Bird\'s Eye View (Original)')
            axes[0, 1].axis('off')
        else:
            axes[0, 1].text(0.5, 0.5, 'No BEV Image', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Bird\'s Eye View (Not Available)')
            axes[0, 1].axis('off')
        
        # 3. BEV 영상 + 바운딩 박스
        if bev_image is not None and len(bev_boxes) > 0:
            bev_with_boxes = draw_boxes_on_bev(bev_image, bev_boxes)
            # axes[1, 0].imshow(cv2.cvtColor(bev_with_boxes, cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title(f'BEV with Bounding Boxes\nDetected: {len(bev_boxes)} objects')
            axes[1, 0].axis('off')
        else:
            if bev_image is not None:
                #axes[1, 0].imshow(cv2.cvtColor(bev_image, cv2.COLOR_BGR2RGB))
                axes[1, 0].set_title('BEV with Bounding Boxes\nNo objects detected')
            else:
                axes[1, 0].text(0.5, 0.5, 'No BEV Image', ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('BEV with Bounding Boxes (Not Available)')
            axes[1, 0].axis('off')
        
        # 4. 차선 정보 및 통계
        info_text = []
        info_text.append(f"Processing Time: {processing_time*1000:.1f} ms")
        info_text.append(f"Steering Angle: {steering_angle:.2f}°")
        info_text.append(f"Speed: {calculated_speed:.2f} m/s")
        
        if lane_info:
            info_text.append(f"Left Lane X: {lane_info.left_x:.1f}")
            info_text.append(f"Right Lane X: {lane_info.right_x:.1f}")
            info_text.append(f"Left Slope: {lane_info.left_slope:.3f}")
            info_text.append(f"Right Slope: {lane_info.right_slope:.3f}")
        
        info_text.append(f"Detected Objects: {len(bev_boxes)}")
        
        axes[1, 1].text(0.1, 0.9, '\n'.join(info_text), transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('System Information')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.pause(0.1)  # 잠시 대기 (FPS 조절)
        
    else:
        # 일반 환경에서 OpenCV 사용
        # BEV 영상에 바운딩 박스 그리기
        if bev_image is not None and len(bev_boxes) > 0:
            bev_with_boxes = draw_boxes_on_bev(bev_image, bev_boxes)
            cv2.imshow('BEV with Bounding Boxes', bev_with_boxes)
        
        # 기존 처리된 이미지 표시
        cv2.imshow('Processed Image', processed_image)
        
        # 정보 텍스트 추가
        info_img = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.putText(info_img, f"Steering: {steering_angle:.1f}°", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(info_img, f"Speed: {calculated_speed:.1f} m/s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(info_img, f"Objects: {len(bev_boxes)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(info_img, f"Time: {processing_time*1000:.1f} ms", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow('System Info', info_img)

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

for name, addr in MOTOR_ADDRESSES.items():
    motors[name] = MMIO(addr, ADDRESS_RANGE)


def load_dpu():
    global dpu, input_data, output_data, shapeIn, shapeOut0, shapeOut1
    
    overlay = DpuOverlay("../dpu/dpu.bit")
    overlay.load_model("../xmodel/tiny-yolov3_coco_256.xmodel")
    
    dpu = overlay.runner
    
    return overlay, dpu

def main():
    overlay = load_dpu()
    controller = DrivingSystemController(overlay, dpu, motors, speed, steering_speed)
    
    # 주차 시스템 초기화
    ultrasonic_sensors = {}
    for name, addr in ULTRASONIC_ADDRESSES.items():
        ultrasonic_sensors[name] = MMIO(addr, ADDRESS_RANGE)
    
    parking_controller = ParkingSystemController(controller.motor_controller, ultrasonic_sensors)
    
    # 시각화 모드 설정
    show_visualization = True
    frame_count = 0
    
    # Jupyter 환경에서 matplotlib 창 설정
    if is_jupyter_environment() and show_visualization:
        plt.figure(figsize=(15, 10))
    
    try:
        # 기존 run 함수를 수정하여 시각화 추가
        video_path = None
        camera_index = 0
        
        # 카메라 또는 비디오 초기화
        if video_path:
            cap = cv2.VideoCapture(video_path)
        else:
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
        print("P: 주차 시스템 모드")
        
        while True:
            if keyboard.is_pressed('1'):
                controller.switch_mode(1)
                break
            elif keyboard.is_pressed('2'):
                controller.switch_mode(2)
                break
            elif keyboard.is_pressed('p'):
                parking_controller.start_parking()
                break
            time.sleep(0.1)

        # 제어 안내 출력
        print("\n키보드 제어 안내:")
        print("Space: 주행 시작/정지")
        print("1/2: 자율주행/수동주행 모드 전환")
        print("P: 주차 시스템 시작/정지")
        print("V: 시각화 켜기/끄기")
        if controller.control_mode == 2:
            print("\n수동 주행 제어:")
            print("W/S: 전진/후진")
            print("A/D: 좌회전/우회전")
            print("R: 긴급 정지")
        print("Q: 프로그램 종료\n")

        # 센서 데이터 출력 시간 제어용 변수
        last_sensor_print_time = time.time()

        while True:
            # 키보드 입력 처리
            if keyboard.is_pressed('space'):
                time.sleep(0.3)  # 디바운싱
                if parking_controller.is_parking_active:
                    print("🚗 주차 시스템이 실행 중입니다. P 키를 눌러 주차를 중지하세요.")
                elif controller.is_running:
                    controller.stop_driving()
                    print("자율주행 중지됨")
                else:
                    controller.start_driving()
                    print("자율주행 시작됨")
            
            elif keyboard.is_pressed('1') or keyboard.is_pressed('2'):
                if parking_controller.is_parking_active:
                    print("주차 시스템이 실행 중입니다. 주차를 먼저 중지하세요.")
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
                    print("자율주행이 실행 중입니다. 주행을 먼저 중지하세요.")
                elif parking_controller.is_parking_active:
                    parking_controller.stop_parking()
                    print("주차 시스템 중지")
                else:
                    parking_controller.start_parking()
                    print("주차 시스템 시작")
                    # 주차 시스템이 시작되면 자동으로 주차 실행 시작
                    print("�� 주차 실행을 시작합니다...")
            
            elif keyboard.is_pressed('v'):
                time.sleep(0.3)  # 디바운싱
                show_visualization = not show_visualization
                print(f"시각화: {'켜짐' if show_visualization else '꺼짐'}")
            
            if keyboard.is_pressed('q'):
                print("\n프로그램을 종료합니다.")
                break

            # 주차 모드에서 초음파 센서 데이터 지속적 출력 (1초마다)
            if parking_controller.is_parking_active:
                current_time = time.time()
                if current_time - last_sensor_print_time >= 1.0:  # 1초마다 출력
                    try:
                        # 센서 데이터 직접 읽기
                        sensor_distances = {}
                        for name, addr in ULTRASONIC_ADDRESSES.items():
                            try:
                                sensor = ultrasonic_sensors[name]
                                distance_raw = sensor.read(0x00, 4)  # 4바이트 읽기
                                distance = int.from_bytes(distance_raw, byteorder='little') / 100.0  # cm 단위로 변환
                                sensor_distances[name] = max(0, min(500, distance))  # 0~500cm 범위로 제한
                            except Exception as e:
                                sensor_distances[name] = 0.0
                                print(f"센서 {name} 읽기 오류: {e}")
                        
                        # 센서 데이터 출력 (주차 시스템에서 사용하는 센서들)
                        print(f"🔍 [초음파 센서] 전방우측: {sensor_distances.get('ultrasonic_0', 0):.1f}cm, "
                              f"중간우측: {sensor_distances.get('ultrasonic_2', 0):.1f}cm, "
                              f"후방우측: {sensor_distances.get('ultrasonic_4', 0):.1f}cm")
                        
                        last_sensor_print_time = current_time
                    except Exception as e:
                        print(f"센서 데이터 읽기 오류: {e}")

            # 주차 시스템 실행 (활성화된 경우)
            if parking_controller.is_parking_active:
                parking_controller.execute_parking_cycle()
                parking_status = parking_controller.get_status()
                print(f"주차 상태: {parking_status['status_message']} (단계: {parking_status['phase']})")
                
                # 주차 시스템이 활성화된 경우 자율주행 처리를 건너뜀
                continue

            # 프레임 처리 (주차 시스템이 비활성화된 경우에만)
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break

            # 이미지 처리 및 차량 제어 (주차 시스템이 비활성화된 경우에만)
            processed_info = controller.process_and_control(frame)
            
            # 시각화
            if show_visualization:
                #visualize_results(processed_info, frame_count)
                frame_count += 1
                
                # Jupyter 환경에서는 프레임 수로 제어
                if is_jupyter_environment() and frame_count > 100:  # 100프레임 후 종료
                    break

    except KeyboardInterrupt:
        print("\n사용자에 의해 중지되었습니다.")
    finally:
        # 리소스 정리
        cap.release()
        if is_jupyter_environment():
            plt.ioff()  # 인터랙티브 모드 비활성화
            plt.close('all')
        else:
            cv2.destroyAllWindows()
        controller.stop_driving()
        parking_controller.stop_parking()
        print("모든 시스템이 정리되었습니다.")

if __name__ == "__main__":
    main()
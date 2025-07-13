#!/usr/bin/env python3
"""
Jupyter 환경에서 BEV 시각화를 테스트하기 위한 스크립트
driving_visualize.py와 동일한 전처리 파이프라인 및 시각화 방식 적용
PYNQ 기반 Jupyter 노트북에서 실행 가능
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import math
from collections import deque
from config import KANAYAMA_CONFIG, HISTORY_CONFIG, anchors, classes_path
from yolo_utils import evaluate, pre_process
from image_processor import ImageProcessor

# 클래스명 로드 함수 추가
with open(classes_path, 'r') as f:
    class_names = f.read().strip().split('\n')

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

def slide_window_search_roi(binary_roi):
    """ROI 내부에서만 슬라이딩 윈도우 검색 (driving_visualize.py와 동일)"""
    nwindows = 15
    window_height = binary_roi.shape[0] // nwindows
    nonzero = binary_roi.nonzero()
    nonzero_y, nonzero_x = nonzero
    margin, minpix = 30, 10
    left_inds, right_inds = [], []
    
    # ROI 내부에서 히스토그램으로 초기 좌표 찾기
    hist = np.sum(binary_roi[binary_roi.shape[0]//2:,:], axis=0)
    mid = hist.shape[0]//2
    left_current = np.argmax(hist[:mid]) if np.max(hist[:mid]) > 0 else mid//2
    right_current = np.argmax(hist[mid:]) + mid if np.max(hist[mid:]) > 0 else mid + mid//2

    for w in range(nwindows):
        y_low = binary_roi.shape[0] - (w+1)*window_height
        y_high = binary_roi.shape[0] - w*window_height
        lx_low, lx_high = max(0, left_current-margin), min(binary_roi.shape[1], left_current+margin)
        rx_low, rx_high = max(0, right_current-margin), min(binary_roi.shape[1], right_current+margin)

        good_l = ((nonzero_y>=y_low)&(nonzero_y<y_high)&
                  (nonzero_x>=lx_low)&(nonzero_x<lx_high)).nonzero()[0]
        good_r = ((nonzero_y>=y_low)&(nonzero_y<y_high)&
                  (nonzero_x>=rx_low)&(nonzero_x<rx_high)).nonzero()[0]

        if len(good_l) > minpix:
            left_current = int(nonzero_x[good_l].mean())
            left_inds.append(good_l)
        if len(good_r) > minpix:
            right_current = int(nonzero_x[good_r].mean())
            right_inds.append(good_r)

    # 직선 피팅 (ROI 좌표계에서)
    if left_inds:
        left_inds = np.concatenate(left_inds)
        leftx, lefty = nonzero_x[left_inds], nonzero_y[left_inds]
        left_fit = np.polyfit(lefty, leftx, 1)
    else:
        left_fit = [0, 0]
    
    if right_inds:
        right_inds = np.concatenate(right_inds)
        rightx, righty = nonzero_x[right_inds], nonzero_y[right_inds]
        right_fit = np.polyfit(righty, rightx, 1)
    else:
        right_fit = [0, 0]

    return left_fit, right_fit

def process_roi(roi, binary_frame, box):
    """ROI 내부 전체 처리 & 직선 피팅 결과 좌표 리턴 (driving_visualize.py와 동일)"""
    # 1. 블러
    blurred = cv2.GaussianBlur(roi, (5,5), 1)
    
    # 2. 추천 전처리 파이프라인 (CLAHE 제외)
    # HLS 색공간 변환
    hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)
    L = hls[:,:,1]  # Lightness
    S = hls[:,:,2]  # Saturation

    # 2-1. Adaptive thresholding on L (CLAHE 제외)
    binary_L = cv2.adaptiveThreshold(L, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=65, C=-10)

    # 2-2. HSV 색상 필터링 조합
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask_hsv = (hsv[:,:,2] > 10) & (hsv[:,:,1] < 180)  # V > 30, S < 120

    # 2-3. 최종 마스크 결합
    final_mask = binary_L & mask_hsv
    
    
    # 이진화 결과를 전체 프레임에 합성
    x, y, x2, y2 = box
    binary_frame[y:y2, x:x2] = final_mask.astype(np.uint8) * 255
    
    # ROI 내부에서만 슬라이딩 윈도우 검색
    left_fit, right_fit = slide_window_search_roi(final_mask.astype(np.uint8) * 255)
    
    return left_fit, right_fit, final_mask.astype(np.uint8) * 255

class LaneInfo:
    """차선 정보를 저장하는 클래스 (driving_visualize.py와 동일)"""
    def __init__(self, w=256):  # 기본값을 256으로 변경
        self.left_x = w // 2  # 영상의 절반으로 초기화
        self.right_x = w // 2  # 영상의 절반으로 초기화
        self.left_slope = 0.0
        self.right_slope = 0.0
        self.left_intercept = 0.0
        self.right_intercept = 0.0
        # fitLine 파라미터 추가
        self.left_params = None   # (vx, vy, x0, y0)
        self.right_params = None  # (vx, vy, x0, y0)
        self.left_points = None  # 슬라이딩 윈도우 결과 저장용
        self.right_points = None

def find_boxes_by_color(frame, lower_hsv, upper_hsv):
    """HSV 범위로 바운딩박스(테두리) 검출 (driving_visualize.py와 동일)"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    # 모폴로지로 테두리만 남기기
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 컨투어로 검출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h > 5000:  # 면적 임계치
            boxes.append((x, y, x+w, y+h))
    return boxes

def extract_lane_info(boxes, frame, binary_frame, color_type, w):
    """차선 정보 추출 - 여러 박스의 픽셀들을 합쳐서 하나의 직선 생성 (driving_visualize.py와 동일)"""
    lane_x = w // 2  # 영상 중앙값으로 초기화
    lane_slope = 0.0
    lane_intercept = 0.0
    
    # 모든 박스에서 추출한 픽셀들을 저장할 리스트
    all_lane_pixels_x = []
    all_lane_pixels_y = []
    
    for box in boxes:
        x, y, x2, y2 = box
        roi = frame[y:y2, x:x2]
        left_fit, right_fit, binary_roi = process_roi(roi, binary_frame, box)
        
        # 노란 박스는 왼쪽 차선, 파란 박스는 오른쪽 차선
        if color_type == "yellow":
            if abs(left_fit[0]) > 1e-6 or abs(left_fit[1]) > 1e-6:
                # ROI 내에서 직선상의 점들을 생성
                roi_height = y2 - y
                roi_y_points = np.linspace(0, roi_height-1, roi_height)
                roi_x_points = left_fit[0] * roi_y_points + left_fit[1]
                
                # 전체 프레임 좌표로 변환
                global_x_points = roi_x_points + x
                global_y_points = roi_y_points + y
                
                # 유효한 점들만 추가 (ROI 범위 내)
                valid_mask = (roi_x_points >= 0) & (roi_x_points < (x2-x))
                all_lane_pixels_x.extend(global_x_points[valid_mask])
                all_lane_pixels_y.extend(global_y_points[valid_mask])
                
        else:  # blue
            if abs(right_fit[0]) > 1e-6 or abs(right_fit[1]) > 1e-6:
                # ROI 내에서 직선상의 점들을 생성
                roi_height = y2 - y
                roi_y_points = np.linspace(0, roi_height-1, roi_height)
                roi_x_points = right_fit[0] * roi_y_points + right_fit[1]
                
                # 전체 프레임 좌표로 변환
                global_x_points = roi_x_points + x
                global_y_points = roi_y_points + y
                
                # 유효한 점들만 추가 (ROI 범위 내)
                valid_mask = (roi_x_points >= 0) & (roi_x_points < (x2-x))
                all_lane_pixels_x.extend(global_x_points[valid_mask])
                all_lane_pixels_y.extend(global_y_points[valid_mask])
    
    # 모든 박스의 픽셀들을 합쳐서 하나의 직선 피팅
    if len(all_lane_pixels_x) > 10:  # 충분한 점이 있을 때만
        # numpy 배열로 변환
        all_x = np.array(all_lane_pixels_x)
        all_y = np.array(all_lane_pixels_y)
        
        # 전체 픽셀들로 직선 피팅 (y = mx + b 형태로)
        combined_fit = np.polyfit(all_y, all_x, 1)
        
        # 프레임 중심에서의 x 좌표 계산
        center_y = frame.shape[0] // 2
        lane_x = combined_fit[0] * center_y + combined_fit[1]
        lane_slope = combined_fit[0]
        lane_intercept = combined_fit[1]
    
    return lane_x, lane_slope, lane_intercept

def generate_left_lane_from_right(right_x, right_slope, right_intercept, frame_width, lane_width_pixels=160):
    """오른쪽 차선을 기준으로 왼쪽 차선을 생성 (driving_visualize.py와 동일)"""
    # 왼쪽 차선은 오른쪽 차선에서 일정 간격만큼 왼쪽에 위치
    left_x = right_x - lane_width_pixels
    
    # 기울기는 오른쪽 차선과 동일 (평행한 차선)
    left_slope = right_slope
    
    # y절편도 동일한 간격만큼 조정
    left_intercept = right_intercept - lane_width_pixels
    
    # 프레임 범위 내로 제한
    left_x = max(0, min(left_x, frame_width - 1))
    
    return left_x, left_slope, left_intercept

def kanayama_control(lane_data, frame_width, Fix_Speed=30, lane_width_m=0.9):
    """driving_visualize.py와 동일한 Kanayama 제어기"""
    # 1) 데이터 없으면 그대로
    if lane_data.left_x == frame_width // 2 and lane_data.right_x == frame_width // 2:
        print("차선을 찾을 수 없습니다.")
        return 0.0, Fix_Speed

    # 2) 픽셀 단위 차로 폭 & 픽셀당 미터 변환 계수
    lane_pixel_width = lane_data.right_x - lane_data.left_x
    if lane_pixel_width > 0:
        pix2m = lane_pixel_width / lane_width_m
    else:
        pix2m = frame_width / lane_width_m  # fallback

    # 3) 횡방향 오차: 차량 중앙(pixel) - 차로 중앙(pixel) → m 단위
    image_cx    = frame_width / 2.0
    lane_cx     = (lane_data.left_x + lane_data.right_x) / 2.0
    lateral_err = (lane_cx - image_cx) / pix2m

    # 4) 헤딩 오차 (차선 기울기 평균)
    heading_err = -0.5 * (lane_data.left_slope + lane_data.right_slope)

    # 5) Kanayama 제어식
    K_y, K_phi, L = 0.1, 0.3, 0.5
    v_r = Fix_Speed
    v = v_r * (math.cos(heading_err))**2
    w = v_r * (K_y * lateral_err + K_phi * math.sin(heading_err))
    delta = math.atan2(w * L, v)

    # 6) 픽셀→도 단위 보정 (k_p)
    steering = math.degrees(delta) * (Fix_Speed/25)
    steering = max(min(steering, 30.0), -30.0)
    
    # 디버깅 정보 출력
    print(f"Lateral error: {lateral_err:.3f}m, Heading error: {math.degrees(heading_err):.1f}°")
    print(f"Lane center: {lane_cx:.1f}, Image center: {image_cx:.1f}")
    print(f"Steering: {steering:.2f}°, Speed: {v:.1f}")
    
    return steering, v

class TestVisualizationController:
    """driving_visualize.py와 동일한 제어 로직을 가진 클래스"""
    
    def __init__(self, frame_width=256):
        # 조향각 히스토리 관리
        self.steering_history = deque(maxlen=HISTORY_CONFIG['max_history_size'])
        self.no_lane_detection_count = 0
        self.max_no_lane_frames = HISTORY_CONFIG['max_no_lane_frames']
        self.default_steering_angle = HISTORY_CONFIG['default_steering_angle']
        self.avg_window_size = HISTORY_CONFIG['avg_window_size']
        self.smoothing_factor = HISTORY_CONFIG['smoothing_factor']
        
        # 영상 가로 크기 저장
        self.frame_width = frame_width
        
        # Kanayama 제어기 파라미터
        self.K_y = KANAYAMA_CONFIG['K_y']
        self.K_phi = KANAYAMA_CONFIG['K_phi']
        self.L = KANAYAMA_CONFIG['L']
        self.lane_width = KANAYAMA_CONFIG['lane_width']
        self.v_r = KANAYAMA_CONFIG['v_r']

    def add_steering_to_history(self, steering_angle):
        """조향각을 히스토리에 추가"""
        self.steering_history.append(steering_angle)
        
    def get_average_steering(self, num_frames=None):
        """최근 N개 조향각의 평균 계산"""
        if len(self.steering_history) == 0:
            return self.default_steering_angle
        
        if num_frames is None:
            num_frames = self.avg_window_size
        
        recent_values = list(self.steering_history)[-min(num_frames, len(self.steering_history)):]
        average_steering = sum(recent_values) / len(recent_values)
        
        return average_steering
        
    def get_smoothed_steering(self, current_steering_angle):
        """스무딩을 적용한 조향각 계산"""
        if len(self.steering_history) == 0:
            return current_steering_angle
        
        previous_angle = self.steering_history[-1]
        smoothed_angle = (self.smoothing_factor * previous_angle + 
                         (1 - self.smoothing_factor) * current_steering_angle)
        
        return smoothed_angle
        
    def should_use_history(self, lane_info):
        """히스토리 사용 여부 결정"""
        if lane_info.left_x == self.frame_width // 2 and lane_info.right_x == self.frame_width // 2:
            self.no_lane_detection_count += 1
            return True
        else:
            self.no_lane_detection_count = 0
            return False
            
    def get_robust_steering_angle(self, lane_info, current_steering_angle):
        """강건한 조향각 계산 (히스토리 + 스무딩 적용)"""
        if self.should_use_history(lane_info):
            if self.no_lane_detection_count <= self.max_no_lane_frames:
                robust_angle = self.get_average_steering()
                print(f"차선 미검출: 이전 {min(self.avg_window_size, len(self.steering_history))}개 값 평균 사용 ({robust_angle:.2f}°)")
                return robust_angle
            else:
                print(f"차선 미검출: 기본값 사용 ({self.default_steering_angle:.2f}°)")
                return self.default_steering_angle
        else:
            smoothed_angle = self.get_smoothed_steering(current_steering_angle)
            self.add_steering_to_history(smoothed_angle)
            
            if len(self.steering_history) > 1:
                print(f"스무딩 적용: {current_steering_angle:.2f}° → {smoothed_angle:.2f}°")
            
            return smoothed_angle

def draw_steering_info_matplotlib(ax, steering_angle, lane_data, visualizer, speed=None):
    """matplotlib을 사용한 조향각과 차선 정보 표시"""
    # 조향각 표시
    ax.text(0.02, 0.95, f"Steering: {steering_angle:.1f}°", 
            transform=ax.transAxes, fontsize=12, color='green', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 속도 정보 표시
    if speed is not None:
        ax.text(0.02, 0.90, f"Speed: {speed:.1f} m/s", 
                transform=ax.transAxes, fontsize=12, color='orange', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 차선 정보 표시
    if lane_data:
        left_status = " (생성됨)" if lane_data.left_x != visualizer.frame_width // 2 and lane_data.left_x < 100 else ""
        ax.text(0.02, 0.85, f"Left X: {lane_data.left_x:.0f}, Slope: {lane_data.left_slope:.3f}{left_status}", 
                transform=ax.transAxes, fontsize=10, color='yellow', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        right_status = " (생성됨)" if lane_data.right_x != visualizer.frame_width // 2 and lane_data.right_x > 156 else ""
        ax.text(0.02, 0.80, f"Right X: {lane_data.right_x:.0f}, Slope: {lane_data.right_slope:.3f}{right_status}", 
                transform=ax.transAxes, fontsize=10, color='red', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 히스토리 정보 표시
    ax.text(0.02, 0.75, f"History size: {len(visualizer.steering_history)}", 
            transform=ax.transAxes, fontsize=10, color='blue', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.text(0.02, 0.70, f"No lane count: {visualizer.no_lane_detection_count}", 
            transform=ax.transAxes, fontsize=10, color='blue', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

def draw_generated_lane_matplotlib(ax, lane_x, lane_slope, lane_intercept, color, is_generated=True):
    """matplotlib을 사용한 생성된 차선 그리기"""
    if abs(lane_slope) < 1e-6 and abs(lane_intercept) < 1e-6:
        return
    
    # 차선의 시작점과 끝점 계산
    y_bottom = 255
    y_top = 128
    
    x_bottom = lane_slope * y_bottom + lane_intercept
    x_top = lane_slope * y_top + lane_intercept
    
    # 프레임 범위 내로 제한
    x_bottom = max(0, min(x_bottom, 255))
    x_top = max(0, min(x_top, 255))
    
    if is_generated:
        # 생성된 차선은 점선으로 표시
        ax.plot([x_bottom, x_top], [y_bottom, y_top], color=color, linestyle='--', linewidth=3, alpha=0.8)
    else:
        # 일반 차선은 실선으로 그리기
        ax.plot([x_bottom, x_top], [y_bottom, y_top], color=color, linewidth=3, alpha=0.8)

class TestVisualizationYOLO:
    def __init__(self, dpu, anchors, class_names):
        self.dpu = dpu
        self.anchors = anchors
        self.class_names = class_names
        inputTensors = dpu.get_input_tensors()
        outputTensors = dpu.get_output_tensors()
        self.shapeIn = tuple(inputTensors[0].dims)
        self.shapeOut0 = tuple(outputTensors[0].dims)
        self.shapeOut1 = tuple(outputTensors[1].dims)
        self.input_data = [np.empty(self.shapeIn, dtype=np.float32, order="C")]
        self.output_data = [
            np.empty(self.shapeOut0, dtype=np.float32, order="C"),
            np.empty(self.shapeOut1, dtype=np.float32, order="C")
        ]

    def infer(self, frame):
        image_data = np.array(pre_process(frame, (256, 256)), dtype=np.float32)
        image_shape = (256, 256)
        self.input_data[0][...] = image_data.reshape(self.shapeIn[1:])
        job_id = self.dpu.execute_async(self.input_data, self.output_data)
        self.dpu.wait(job_id)
        conv_out0 = np.reshape(self.output_data[0], self.shapeOut0)
        conv_out1 = np.reshape(self.output_data[1], self.shapeOut1)
        yolo_outputs = [conv_out0, conv_out1]
        boxes, scores, classes = evaluate(yolo_outputs, image_shape, self.class_names, self.anchors)
        return boxes, scores, classes

def test_single_frame_visualization(dpu, camera_index=0, output_video_path="output_lane_detection.mp4", max_frames=100, save_png_interval=0.5, png_output_dir="output_frames"):
    """실시간 카메라 + 동영상 저장 + BEV 변환 + PNG 저장"""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"카메라 {camera_index}를 열 수 없습니다.")
        return
    
    # 카메라 해상도 설정 (640x480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f"실시간 카메라 모드로 실행 중... (최대 {max_frames}프레임)")
    print(f"카메라 해상도: 640x480")
    print(f"동영상 저장 경로: {output_video_path}")
    print(f"PNG 저장 간격: {save_png_interval}초")
    print(f"PNG 저장 디렉토리: {png_output_dir}")
    
    # PNG 저장 디렉토리 생성
    import os
    os.makedirs(png_output_dir, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10
    frame_size = (256*2, 256)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    
    frame_count = 0
    last_png_save_time = 0
    visualizer = TestVisualizationController(frame_width=256)
    yolo = TestVisualizationYOLO(dpu, anchors, class_names)
    
    # BEV 변환용 예시 좌표 (image_processor.py와 동일한 방식으로 수정)
    srcmat = np.float32([[250, 316], [380, 316], [450, 476], [200, 476]])
    # 이미지 크기에 비례하여 dstmat 설정 (image_processor.py와 동일)
    frame_size = 256  # BEV 변환 후 이미지 크기
    dstmat = np.float32([
        [round(frame_size * 0.3), 0],                    # 좌상단
        [round(frame_size * 0.7), 0],                    # 우상단
        [round(frame_size * 0.7), frame_size],           # 우하단
        [round(frame_size * 0.3), frame_size]            # 좌하단
    ])
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("카메라에서 프레임을 읽을 수 없습니다.")
            break
            
        current_time = time.time()
        
        # BEV 변환
        frame_bev = ImageProcessor(dpu, classes_path, anchors).bird_convert(frame, srcmat, dstmat)
        
        # 이하 기존 파이프라인에서 frame -> frame_bev로 대체
        boxes, scores, classes = yolo.infer(frame_bev)
        left_lane_boxes = [tuple(map(int, box)) for box, cls in zip(boxes, classes) if cls == 0]
        right_lane_boxes = [tuple(map(int, box)) for box, cls in zip(boxes, classes) if cls == 1]
        
        frame_bev = cv2.resize(frame_bev, (256, 256))
        h, w = frame_bev.shape[:2]
        binary_frame = np.zeros((h, w), dtype=np.uint8)
        lane_data = LaneInfo(w)
        
        left_x, left_slope, left_intercept = extract_lane_info(left_lane_boxes, frame_bev, binary_frame, "yellow", w)
        lane_data.left_x = left_x
        lane_data.left_slope = left_slope
        lane_data.left_intercept = left_intercept
        
        right_x, right_slope, right_intercept = extract_lane_info(right_lane_boxes, frame_bev, binary_frame, "blue", w)
        lane_data.right_x = right_x
        lane_data.right_slope = right_slope
        lane_data.right_intercept = right_intercept
        
        if lane_data.left_x == w // 2 and lane_data.right_x != w // 2:
            left_x, left_slope, left_intercept = generate_left_lane_from_right(
                lane_data.right_x, lane_data.right_slope, lane_data.right_intercept, w, lane_width_pixels=160
            )
            lane_data.left_x = left_x
            lane_data.left_slope = left_slope
            lane_data.left_intercept = left_intercept
        elif lane_data.right_x == w // 2 and lane_data.left_x != w // 2:
            right_x = lane_data.left_x + 160
            right_slope = lane_data.left_slope
            right_intercept = lane_data.left_intercept + 160
            right_x = max(0, min(right_x, w - 1))
            lane_data.right_x = right_x
            lane_data.right_slope = right_slope
            lane_data.right_intercept = right_intercept
            
        frame_w = frame_bev.shape[1]
        base_steering_angle, speed = kanayama_control(lane_data, frame_w)
        steering_angle = visualizer.get_robust_steering_angle(lane_data, base_steering_angle)
        
        # 차선 박스 그리기
        for box in left_lane_boxes:
            x, y, x2, y2 = box
            cv2.rectangle(frame_bev, (x, y), (x2, y2), (0, 255, 255), 2)
        for box in right_lane_boxes:
            x, y, x2, y2 = box
            cv2.rectangle(frame_bev, (x, y), (x2, y2), (255, 0, 0), 2)
            
        # 차선 그리기
        if lane_data.left_slope != 0.0 or lane_data.left_intercept != 0.0:
            y_bottom = h - 1
            y_top = h // 2
            x_bottom = lane_data.left_slope * y_bottom + lane_data.left_intercept
            x_top = lane_data.left_slope * y_top + lane_data.left_intercept
            x_bottom = max(0, min(x_bottom, w - 1))
            x_top = max(0, min(x_top, w - 1))
            color = (0, 255, 0) if lane_data.left_x < 100 else (0, 255, 255)
            cv2.line(frame_bev, (int(x_bottom), int(y_bottom)), (int(x_top), int(y_top)), color, 2)
            
        if lane_data.right_slope != 0.0 or lane_data.right_intercept != 0.0:
            y_bottom = h - 1
            y_top = h // 2
            x_bottom = lane_data.right_slope * y_bottom + lane_data.right_intercept
            x_top = lane_data.right_slope * y_top + lane_data.right_intercept
            x_bottom = max(0, min(x_bottom, w - 1))
            x_top = max(0, min(x_top, w - 1))
            color = (0, 0, 255) if lane_data.right_x > 156 else (255, 0, 0)
            cv2.line(frame_bev, (int(x_bottom), int(y_bottom)), (int(x_top), int(y_top)), color, 2)
            
        # 텍스트 정보 추가
        cv2.putText(frame_bev, f"Steering: {steering_angle:.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_bev, f"Speed: {speed:.1f} m/s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame_bev, f"Frame: {frame_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame_bev, f"Time: {current_time:.1f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # 조향 화살표 그리기
        center_x, center_y = w // 2, h - 30
        arrow_length = int(abs(steering_angle) * 2)
        if steering_angle > 0:
            end_x = center_x + arrow_length
            cv2.arrowedLine(frame_bev, (center_x, center_y), (end_x, center_y), (0, 0, 255), 3)
            cv2.putText(frame_bev, "RIGHT", (center_x + 20, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif steering_angle < 0:
            end_x = center_x - arrow_length
            cv2.arrowedLine(frame_bev, (center_x, center_y), (end_x, center_y), (0, 0, 255), 3)
            cv2.putText(frame_bev, "LEFT", (center_x - 70, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.arrowedLine(frame_bev, (center_x, center_y), (center_x, center_y - 30), (0, 255, 0), 3)
            cv2.putText(frame_bev, "STRAIGHT", (center_x - 40, center_y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        # 이진화 영상에 박스 그리기
        binary_3ch = cv2.cvtColor(binary_frame, cv2.COLOR_GRAY2BGR)
        for box in left_lane_boxes:
            x, y, x2, y2 = box
            cv2.rectangle(binary_3ch, (x, y), (x2, y2), (0, 255, 255), 2)
        for box in right_lane_boxes:
            x, y, x2, y2 = box
            cv2.rectangle(binary_3ch, (x, y), (x2, y2), (255, 0, 0), 2)
            
        # 결합된 프레임 생성
        combined_frame = np.hstack([frame_bev, binary_3ch])
        
        # 동영상 저장
        out.write(combined_frame)
        
        # PNG 저장 (0.5초 간격)
        if current_time - last_png_save_time >= save_png_interval:
            png_filename = os.path.join(png_output_dir, f"frame_{frame_count:04d}_time_{current_time:.1f}s.png")
            cv2.imwrite(png_filename, combined_frame)
            print(f"PNG 저장: {png_filename}")
            last_png_save_time = current_time
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"처리된 프레임: {frame_count}")
            
    cap.release()
    out.release()
    print(f"총 {frame_count}개 프레임 처리 완료")
    print(f"동영상 저장 완료: {output_video_path}")
    print(f"PNG 파일들 저장 완료: {png_output_dir}/")
    print(f"평균 조향각: {visualizer.get_average_steering():.2f}°")

def test_video_visualization(dpu, max_frames=30, camera_index=0):
    """비디오 시각화 테스트 (YOLO 박스 기반, BEV 변환, 실시간 카메라만 지원)"""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"카메라 {camera_index}를 열 수 없습니다.")
        return
    
    # 카메라 해상도 설정 (640x480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("실시간 카메라 모드로 실행 중... (ESC 키로 종료)")
    print(f"카메라 해상도: 640x480")
    frame_count = 0
    visualizer = TestVisualizationController(frame_width=256)
    steering_angles = []
    speeds = []
    frame_numbers = []
    yolo = TestVisualizationYOLO(dpu, anchors, class_names)
    # image_processor.py와 동일한 BEV 변환 좌표 설정
    srcmat = np.float32([[250, 316], [380, 316], [450, 476], [200, 476]])
    frame_size = 256  # BEV 변환 후 이미지 크기
    dstmat = np.float32([
        [round(frame_size * 0.3), 0],                    # 좌상단
        [round(frame_size * 0.7), 0],                    # 우상단
        [round(frame_size * 0.7), frame_size],           # 우하단
        [round(frame_size * 0.3), frame_size]            # 좌하단
    ])
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_bev = ImageProcessor(dpu, classes_path, anchors).bird_convert(frame, srcmat, dstmat)
        boxes, scores, classes = yolo.infer(frame_bev)
        left_lane_boxes = [tuple(map(int, box)) for box, cls in zip(boxes, classes) if cls == 0]
        right_lane_boxes = [tuple(map(int, box)) for box, cls in zip(boxes, classes) if cls == 1]
        frame_bev = cv2.resize(frame_bev, (256, 256))
        h, w = frame_bev.shape[:2]
        binary_frame = np.zeros((h, w), dtype=np.uint8)
        lane_data = LaneInfo(w)
        left_x, left_slope, left_intercept = extract_lane_info(left_lane_boxes, frame_bev, binary_frame, "yellow", w)
        lane_data.left_x = left_x
        lane_data.left_slope = left_slope
        lane_data.left_intercept = left_intercept
        right_x, right_slope, right_intercept = extract_lane_info(right_lane_boxes, frame_bev, binary_frame, "blue", w)
        lane_data.right_x = right_x
        lane_data.right_slope = right_slope
        lane_data.right_intercept = right_intercept
        if lane_data.left_x == w // 2 and lane_data.right_x != w // 2:
            left_x, left_slope, left_intercept = generate_left_lane_from_right(
                lane_data.right_x, lane_data.right_slope, lane_data.right_intercept, w, lane_width_pixels=160
            )
            lane_data.left_x = left_x
            lane_data.left_slope = left_slope
            lane_data.left_intercept = left_intercept
        elif lane_data.right_x == w // 2 and lane_data.left_x != w // 2:
            right_x = lane_data.left_x + 160
            right_slope = lane_data.left_slope
            right_intercept = lane_data.left_intercept + 160
            right_x = max(0, min(right_x, w - 1))
            lane_data.right_x = right_x
            lane_data.right_slope = right_slope
            lane_data.right_intercept = right_intercept
        frame_w = frame_bev.shape[1]
        base_steering_angle, speed = kanayama_control(lane_data, frame_w)
        steering_angle = visualizer.get_robust_steering_angle(lane_data, base_steering_angle)
        steering_angles.append(steering_angle)
        speeds.append(speed)
        frame_numbers.append(frame_count)
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"처리된 프레임: {frame_count}")
    cap.release()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    ax1 = axes[0]
    ax1.plot(frame_numbers, steering_angles, 'b-', linewidth=2, label='Steering Angle')
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Steering Angle (degrees)')
    ax1.set_title('Camera Steering Angle Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2 = axes[1]
    ax2.plot(frame_numbers, speeds, 'r-', linewidth=2, label='Speed')
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Speed (m/s)')
    ax2.set_title('Camera Speed Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    plt.show()
    print(f"총 {frame_count}개 프레임 처리 완료")
    print(f"평균 조향각: {np.mean(steering_angles):.2f}°")
    print(f"평균 속도: {np.mean(speeds):.2f} m/s")

if __name__ == "__main__":
    print("Jupyter 환경에서 실행 중...")
    print("사용 가능한 함수들:")
    print("1. test_single_frame_visualization(dpu, camera_index=0, output_video_path='output.mp4', max_frames=100)")
    print("   - 실시간 카메라 + 동영상 저장 (OpenCV)")
    print()
    print("2. test_video_visualization(dpu, max_frames=30, camera_index=0)")
    print("   - 비디오 시각화 (matplotlib 그래프, 실시간 카메라)")
    print()
    print("사용 예시:")
    print("# 실시간 카메라 + 동영상 저장 (100프레임)")
    print("test_single_frame_visualization(dpu, camera_index=0, output_video_path='lane_detection.mp4', max_frames=100)")
    print()
    print("# 비디오 (카메라, 50프레임)")
    print("test_video_visualization(dpu, max_frames=50, camera_index=0)") 
# Copyright (c) 2024 Sungkyunkwan University AutomationLab
#
# Authors:
# - Gyuhyeon Hwang <rbgus7080@g.skku.edu>, Hobin Oh <hobin0676@daum.net>, Minkwan Choi <arbong97@naver.com>, Hyeonjin Sim <nufxwms@naver.com>
# - url: https://micro.skku.ac.kr/micro/index.do

import cv2
import numpy as np
import math
import os
import colorsys
import random
from PIL import Image
import time
from collections import deque
from yolo_utils import pre_process, evaluate
from config import KANAYAMA_CONFIG, HISTORY_CONFIG

def slide_window_in_roi(binary, box, n_win=15, margin=30, minpix=10):
    """
    debugging/visualize.py의 slide_window_search_roi와 동일하게 슬라이딩 윈도우 적용
    화면 하단 10%만 사용
    
    
    Args:
        binary: 2D np.array (전체 BEV 이진화 이미지)
        box: (y1, x1, y2, x2) – 전체 이미지 좌표계
        n_win: 윈도우 개수 (15)
        margin: 윈도우 마진 (30)
        minpix: 최소 픽셀 수 (10)
    Returns:
        fit: (slope, intercept), lane_pts: (x, y) 리스트
    """
    y1, x1, y2, x2 = box
    roi = binary[int(y1):int(y2), int(x1):int(x2)]
    if roi.size == 0:
        return None, None

    # 화면 하단 10%만 사용하기 위한 y 좌표 필터링
    roi_height = roi.shape[0]
    use_bottom_10_percent = int(roi_height * 0.2)  # 하단 10% 시작점
    
    window_height = roi.shape[0] // n_win
    nonzero = roi.nonzero()
    nonzero_y, nonzero_x = nonzero
    
    # 하단 10%의 픽셀들만 사용
    valid_mask = nonzero_y >= use_bottom_10_percent
    nonzero_y = nonzero_y[valid_mask]
    nonzero_x = nonzero_x[valid_mask]
    
    left_inds = []

    # ROI 내부에서 히스토그램으로 초기 좌표 찾기 (하단 10%만 사용)
    hist = np.sum(roi[use_bottom_10_percent:,:], axis=0)
    if np.max(hist) > 0:
        current_x = np.argmax(hist)
    else:
        current_x = roi.shape[1] // 2

    for w in range(n_win):
        y_low = roi.shape[0] - (w+1)*window_height
        y_high = roi.shape[0] - w*window_height
        
        # 하단 10% 영역만 사용
        if y_high <= use_bottom_10_percent:
            continue
            
        # y_low도 하단 10% 영역 내에 있도록 조정
        y_low = max(y_low, use_bottom_10_percent)
        
        x_low = max(0, current_x-margin)
        x_high = min(roi.shape[1], current_x+margin)

        good_inds = ((nonzero_y>=y_low)&(nonzero_y<y_high)&(nonzero_x>=x_low)&(nonzero_x<x_high)).nonzero()[0]
        if len(good_inds) > minpix:
            current_x = int(nonzero_x[good_inds].mean())
            left_inds.append(good_inds)

    if left_inds:
        left_inds = np.concatenate(left_inds)
        leftx, lefty = nonzero_x[left_inds], nonzero_y[left_inds]
        # ROI 좌표를 전체 이미지 좌표로 변환
        leftx_global = leftx + int(x1)
        lefty_global = lefty + int(y1)
        if len(leftx_global) >= 2:
            left_fit = np.polyfit(lefty_global, leftx_global, 1)
            return left_fit, (leftx_global, lefty_global)
    return None, None

class LaneInfo:
    """차선 정보를 저장하는 클래스"""
    def __init__(self, w=256):
        self.left_x = w // 2  # 왼쪽 차선 x좌표 (기본값: 영상 중앙)
        self.right_x = w // 2  # 오른쪽 차선 x좌표 (기본값: 영상 중앙)
        self.left_slope = 0.0  # 왼쪽 차선 기울기
        self.left_intercept = 0.0  # 왼쪽 차선 y절편
        self.right_slope = 0.0  # 오른쪽 차선 기울기
        self.right_intercept = 0.0  # 오른쪽 차선 y절편
        # fitLine 파라미터 추가
        self.left_params = None   # (vx, vy, x0, y0)
        self.right_params = None  # (vx, vy, x0, y0)
        self.left_points = None  # 슬라이딩 윈도우 결과 저장용
        self.right_points = None

class ImageProcessor:
    def __init__(self, dpu, classes_path, anchors, parking_mode=False):
        # 클래스 변수로 저장
        self.dpu = dpu
            
        self.classes_path = classes_path
        self.anchors = anchors
        self.class_names = self.load_classes(classes_path)    
        
        self.reference_point_x = 200
        self.reference_point_y = 240
        self.point_detection_height = 20
        
        # 주차모드 플래그 추가
        self.parking_mode = parking_mode
        
        # DPU 초기화 상태 추적 플래그
        self.initialized = False
        self.init_dpu()
        
        # Kanayama 제어기 파라미터 (config에서 불러오기)
        self.K_y = KANAYAMA_CONFIG['K_y']
        self.K_phi = KANAYAMA_CONFIG['K_phi']
        self.L = KANAYAMA_CONFIG['L']
        self.lane_width = KANAYAMA_CONFIG['lane_width']
        self.v_r = KANAYAMA_CONFIG['v_r']
        
        # 조향각 히스토리 관리 (config에서 불러오기)
        self.steering_history = deque(maxlen=HISTORY_CONFIG['max_history_size'])
        self.no_lane_detection_count = 0
        self.max_no_lane_frames = HISTORY_CONFIG['max_no_lane_frames']
        self.default_steering_angle = HISTORY_CONFIG['default_steering_angle']
        self.avg_window_size = HISTORY_CONFIG['avg_window_size']
        self.smoothing_factor = HISTORY_CONFIG['smoothing_factor']
    
    def _log(self, msg):
        if not self.parking_mode:
            print(msg)
    
    def load_classes(self, classes_path):
        """Load class names from the given path"""
        with open(classes_path, 'r') as f:
            class_names = f.read().strip().split('\n')
        return class_names 
    
    def init_dpu(self):
        """DPU 초기화 - 한 번만 실행됨"""
        if self.initialized:
            self._log("DPU 이미 초기화됨")
            return  # 이미 초기화되었으면 바로 리턴

        self._log("DPU 초기화 중...")
        inputTensors = self.dpu.get_input_tensors()
        outputTensors = self.dpu.get_output_tensors()

        self.shapeIn = tuple(inputTensors[0].dims)
        self.shapeOut0 = tuple(outputTensors[0].dims)
        self.shapeOut1 = tuple(outputTensors[1].dims)

        outputSize0 = int(outputTensors[0].get_data_size() / self.shapeIn[0])
        outputSize1 = int(outputTensors[1].get_data_size() / self.shapeIn[0])

        self.input_data = [np.empty(self.shapeIn, dtype=np.float32, order="C")]
        self.output_data = [
            np.empty(self.shapeOut0, dtype=np.float32, order="C"),
            np.empty(self.shapeOut1, dtype=np.float32, order="C")
        ]

        # 초기화 완료 플래그 설정
        self.initialized = True
        self._log("DPU 초기화 완료")

    
    def roi_rectangle_below(self, img, cutting_idx):
        return img[cutting_idx:, :]

    def warpping(self, image, srcmat, dstmat):
        h, w = image.shape[0], image.shape[1]
        transform_matrix = cv2.getPerspectiveTransform(srcmat, dstmat)
        minv = cv2.getPerspectiveTransform(dstmat, srcmat)
        _image = cv2.warpPerspective(image, transform_matrix, (w, h))
        return _image, minv
    
    def bird_convert(self, img, srcmat, dstmat):
        srcmat = np.float32(srcmat)
        dstmat = np.float32(dstmat)
        img_warpped, _ = self.warpping(img, srcmat, dstmat)
        return img_warpped

    def calculate_angle(self, x1, y1, x2, y2):
        if x1 == x2:
            return 90.0
        slope = (y2 - y1) / (x2 - x1)
        return -math.degrees(math.atan(slope))

    def color_filter(self, image):
        """참고 코드와 동일한 흰색 픽셀 검출 방식"""
        # HSV 색공간으로 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 흰색 픽셀 검출 (참고 코드와 동일한 방식)
        lower = np.array([0, 255, 255])  # 흰색 필터
        upper = np.array([255, 255, 255])
        white_mask = cv2.inRange(hsv, lower, upper)
        masked = cv2.bitwise_and(image, image, mask=white_mask)
        
        return masked

    def generate_left_lane_from_right(self, right_x, right_slope, right_intercept, frame_width, lane_width_pixels=200):
        """
        오른쪽 차선을 기준으로 왼쪽 차선을 생성
        Args:
            right_x: 오른쪽 차선의 x 좌표
            right_slope: 오른쪽 차선의 기울기
            right_intercept: 오른쪽 차선의 y절편
            frame_width: 프레임 너비
            lane_width_pixels: 차선 간격 (픽셀)
        Returns:
            left_x, left_slope, left_intercept: 생성된 왼쪽 차선 정보
        """
        # 왼쪽 차선은 오른쪽 차선에서 일정 간격만큼 왼쪽에 위치
        left_x = right_x - lane_width_pixels
        
        # 기울기는 오른쪽 차선과 동일 (평행한 차선)
        left_slope = right_slope
        
        # y절편도 동일한 간격만큼 조정
        left_intercept = right_intercept - lane_width_pixels
        
        # 프레임 범위 내로 제한
        left_x = max(0, min(left_x, frame_width - 1))
        
        return left_x, left_slope, left_intercept

    def extract_lane_info_improved(self, xyxy_results, classes_results, processed_img):
        """여러 바운딩 박스의 픽셀들을 합쳐서 하나의 직선 생성 (driving_visualize.py와 동일)"""
        h, w = processed_img.shape[:2]
        lane_info = LaneInfo(w)
        
        if len(xyxy_results) == 0:
            return lane_info
            
        # 왼쪽 차선과 오른쪽 차선을 위한 픽셀 저장 리스트
        left_lane_pixels_x = []
        left_lane_pixels_y = []
        right_lane_pixels_x = []
        right_lane_pixels_y = []
        
        for i, box in enumerate(xyxy_results):
            y1, x1, y2, x2 = [int(v) for v in box]
            
            # 바운딩 박스 좌표가 이미지 범위를 벗어나는 경우 처리
            y1 = max(0, min(y1, processed_img.shape[0] - 1))
            x1 = max(0, min(x1, processed_img.shape[1] - 1))
            y2 = max(0, min(y2, processed_img.shape[0]))
            x2 = max(0, min(x2, processed_img.shape[1]))
            
            # 유효한 ROI 영역인지 확인
            if y1 >= y2 or x1 >= x2:
                continue
                
            roi = processed_img[y1:y2, x1:x2]
            
            # ROI가 비어있거나 유효하지 않은 경우 건너뛰기
            if roi is None or roi.size == 0:
                continue
                
            # ROI 크기가 너무 작은 경우도 건너뛰기 (최소 5x5 픽셀)
            if roi.shape[0] < 5 or roi.shape[1] < 5:
                continue
                
            # === driving_visualize.py의 process_roi와 동일한 전처리 파이프라인 ===
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
            mask_hsv = (hsv[:,:,2] > 10) & (hsv[:,:,1] < 180)  # V > 10, S < 180

            # 2-3. 최종 마스크 결합
            final_mask = binary_L & mask_hsv

            # 전체 이미지 크기의 이진화 마스크 생성
            full_binary_mask = np.zeros((h, w), dtype=np.uint8)
            full_binary_mask[y1:y2, x1:x2] = final_mask.astype(np.uint8) * 255

            # 슬라이딩 윈도우 적용 (전체 이미지 마스크 사용)
            fit, pts = slide_window_in_roi(full_binary_mask, (y1, x1, y2, x2), n_win=15, margin=30, minpix=10)
            
            if fit is not None and pts is not None:
                slope, intercept = fit
                xs, ys = pts
                
                # 클래스 기반 좌우 분류
                class_id = classes_results[i] if i < len(classes_results) else 0
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else ""
                
                if "left" in class_name.lower():
                    # 왼쪽 차선 픽셀 추가
                    left_lane_pixels_x.extend(xs)
                    left_lane_pixels_y.extend(ys)
                elif "right" in class_name.lower():
                    # 오른쪽 차선 픽셀 추가
                    right_lane_pixels_x.extend(xs)
                    right_lane_pixels_y.extend(ys)
                else:
                    # 클래스가 명확하지 않은 경우 이미지 중앙 기준으로 분류
                    image_center_x = w / 2
                    x_bottom = slope * (h - 1) + intercept
                    if x_bottom < image_center_x:
                        left_lane_pixels_x.extend(xs)
                        left_lane_pixels_y.extend(ys)
                    else:
                        right_lane_pixels_x.extend(xs)
                        right_lane_pixels_y.extend(ys)
        
        # 왼쪽 차선 픽셀들로 직선 피팅
        if len(left_lane_pixels_x) > 10:  # 충분한 점이 있을 때만
            left_x = np.array(left_lane_pixels_x)
            left_y = np.array(left_lane_pixels_y)
            
            # 화면 하단 10%의 픽셀들만 사용
            h = processed_img.shape[0]
            use_bottom_10_percent = int(h * 0.5)
            valid_mask = left_y >= use_bottom_10_percent
            left_x = left_x[valid_mask]
            left_y = left_y[valid_mask]
            
            # 충분한 점이 남아있는지 확인
            if len(left_x) > 5:
                # 전체 픽셀들로 직선 피팅 (y = mx + b 형태로)
                left_fit = np.polyfit(left_y, left_x, 1)
                
                # 프레임 중심에서의 x 좌표 계산
                center_y = h // 2
                lane_info.left_x = left_fit[0] * center_y + left_fit[1]
                lane_info.left_slope = left_fit[0]
                lane_info.left_intercept = left_fit[1]
        
        # 오른쪽 차선 픽셀들로 직선 피팅
        if len(right_lane_pixels_x) > 10:  # 충분한 점이 있을 때만
            right_x = np.array(right_lane_pixels_x)
            right_y = np.array(right_lane_pixels_y)
            
            # 화면 하단 10%의 픽셀들만 사용
            h = processed_img.shape[0]
            use_bottom_10_percent = int(h * 0.5)
            valid_mask = right_y >= use_bottom_10_percent
            right_x = right_x[valid_mask]
            right_y = right_y[valid_mask]
            
            # 충분한 점이 남아있는지 확인
            if len(right_x) > 5:
                # 전체 픽셀들로 직선 피팅 (y = mx + b 형태로)
                right_fit = np.polyfit(right_y, right_x, 1)
                
                # 프레임 중심에서의 x 좌표 계산
                center_y = h // 2
                lane_info.right_x = right_fit[0] * center_y + right_fit[1]
                lane_info.right_slope = right_fit[0]
                lane_info.right_intercept = right_fit[1]
        
        return lane_info

    
    def kanayama_control(self, lane_info):
        """debugging/visualize.py와 동일한 Kanayama 제어기"""
        # 이미지 크기 (256x256으로 리사이즈됨)
        frame_width = 256
        
        # 1) 데이터 없으면 그대로
        if lane_info.left_x == frame_width // 2 and lane_info.right_x == frame_width // 2:
            self._log("차선을 찾을 수 없습니다.")
            # 히스토리에서 평균값 사용
            if len(self.steering_history) > 0:
                avg_steering = self.get_average_steering()
                self._log(f"히스토리 평균 조향각 사용: {avg_steering:.2f}°")
                return avg_steering, self.v_r
            else:
                return self.default_steering_angle, self.v_r
        
        lane_width_m = 0.9  # debugging/visualize.py와 동일한 차로 폭
        Fix_Speed = self.v_r
        
        # 2) 픽셀 단위 차로 폭 & 픽셀당 미터 변환 계수
        lane_pixel_width = lane_info.right_x - lane_info.left_x
        if lane_pixel_width > 0:
            pix2m = lane_pixel_width / lane_width_m
        else:
            pix2m = frame_width / lane_width_m  # fallback

        # 3) 횡방향 오차: 차량 중앙(pixel) - 차로 중앙(pixel) → m 단위
        image_cx = frame_width / 2.0
        lane_cx = (lane_info.left_x + lane_info.right_x) / 2.0
        lateral_err = (lane_cx - image_cx) / pix2m

        # 4) 헤딩 오차 (차선 기울기 평균)
        heading_err = -0.1 * (lane_info.left_slope + lane_info.right_slope)

        # 5) Kanayama 제어식 (debugging/visualize.py와 동일한 파라미터)
        K_y, K_phi, L = 0.3, 0.9, 0.5
        v_r = Fix_Speed
        v = v_r * (math.cos(heading_err))**2
        w = v_r * (K_y * lateral_err + K_phi * math.sin(heading_err))
        delta = math.atan2(w * L, v)

        # 6) 픽셀→도 단위 보정 (k_p) - debugging/visualize.py와 동일
        steering = math.degrees(delta) * (Fix_Speed/25)
        steering = max(min(steering, 30.0), -30.0)
        
        # 디버깅 정보 출력
        self._log(f"Lateral error: {lateral_err:.3f}m, Heading error: {math.degrees(heading_err):.1f}°")
        self._log(f"Lane center: {lane_cx:.1f}, Image center: {image_cx:.1f}")
        self._log(f"Steering: {steering:.2f}°, Speed: {v:.1f}")
        self._log("")  # 빈 줄 추가
        
        return steering, v

    def add_steering_to_history(self, steering_angle):
        """조향각을 히스토리에 추가"""
        self.steering_history.append(steering_angle)
        
    def get_average_steering(self, num_frames=None):
        """최근 N개 조향각의 평균 계산"""
        if len(self.steering_history) == 0:
            return self.default_steering_angle
        
        # 기본값 사용
        if num_frames is None:
            num_frames = self.avg_window_size
        
        # 최근 N개 값만 사용
        recent_values = list(self.steering_history)[-min(num_frames, len(self.steering_history)):]
        
        # 평균 계산
        average_steering = sum(recent_values) / len(recent_values)
        
        return average_steering
        
    def get_smoothed_steering(self, current_steering_angle):
        """스무딩을 적용한 조향각 계산"""
        if len(self.steering_history) == 0:
            return current_steering_angle
        
        # 이전 값과 현재 값을 가중 평균
        previous_angle = self.steering_history[-1]
        smoothed_angle = (self.smoothing_factor * previous_angle + 
                         (1 - self.smoothing_factor) * current_steering_angle)
        
        return smoothed_angle
        
    def should_use_history(self, lane_info):
        """히스토리 사용 여부 결정"""
        # 이미지 크기 (256x256으로 리사이즈됨)
        frame_width = 256
        
        # 양쪽 차선이 모두 보이지 않는 경우
        if lane_info.left_x == frame_width // 2 and lane_info.right_x == frame_width // 2:
            self.no_lane_detection_count += 1
            return True
        else:
            # 차선이 보이면 카운터 리셋
            self.no_lane_detection_count = 0
            return False
            
    def get_robust_steering_angle(self, lane_info, current_steering_angle):
        """강건한 조향각 계산 (히스토리 + 스무딩 적용)"""
        if self.should_use_history(lane_info):
            if self.no_lane_detection_count <= self.max_no_lane_frames:
                # 이전 값들의 평균 사용
                robust_angle = self.get_average_steering()
                self._log(f"차선 미검출: 이전 {min(self.avg_window_size, len(self.steering_history))}개 값 평균 사용 ({robust_angle:.2f}°)")
                return robust_angle
            else:
                # 너무 오래 차선을 못 찾으면 기본값 사용
                self._log(f"차선 미검출: 기본값 사용 ({self.default_steering_angle:.2f}°)")
                return self.default_steering_angle
        else:
            # 차선이 보이면 스무딩 적용
            smoothed_angle = self.get_smoothed_steering(current_steering_angle)
            
            # 히스토리에 추가 (스무딩된 값)
            self.add_steering_to_history(smoothed_angle)
            
            # 스무딩이 적용되었는지 출력
            if len(self.steering_history) > 1:
                self._log(f"스무딩 적용: {current_steering_angle:.2f}° → {smoothed_angle:.2f}°")
            
            return smoothed_angle

    def process_frame(self, img):
        """프레임 처리 및 조향각 계산"""
        h, w = img.shape[0], img.shape[1]
        dst_mat = [[round(w * 0.3), 0], [round(w * 0.7), 0], 
                  [round(w * 0.7), h], [round(w * 0.3), h]]
        src_mat = [[238, 276], [382, 276], [450, 436], [200, 436]]
        
        # BEV 변환
        bird_img = self.bird_convert(img, srcmat=src_mat, dstmat=dst_mat)
        
        # 원본 BEV 영상 저장 (바운딩 박스 그리기 전)
        original_bev = bird_img.copy()
        
        img = cv2.resize(bird_img, (256, 256))
        image_size = img.shape[:2]
        image_data = np.array(pre_process(img, (256, 256)), dtype=np.float32)
        
        start_time = time.time()

        # self를 사용하여 클래스 변수에 접근
        self.input_data[0][...] = image_data.reshape(self.shapeIn[1:])
        job_id = self.dpu.execute_async(self.input_data, self.output_data)
        self.dpu.wait(job_id)
        end_time = time.time()
        
        conv_out0 = np.reshape(self.output_data[0], self.shapeOut0)
        conv_out1 = np.reshape(self.output_data[1], self.shapeOut1)
        yolo_outputs = [conv_out0, conv_out1]

        boxes, scores, classes = evaluate(yolo_outputs, image_size, self.class_names, self.anchors)

        # 바운딩 박스 시각화 (디버깅용)
        for i, box in enumerate(boxes):
            top_left = (int(box[1]), int(box[0]))
            bottom_right = (int(box[3]), int(box[2]))
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

        # === 바운딩 박스 기반 차선(직선) 방정식 시각화 추가 ===
        lane_info = self.extract_lane_info_improved(boxes, classes, img)
        h, w = img.shape[:2]
        
        # 왼쪽 차선이 검출되지 않았을 때 오른쪽 차선을 기준으로 왼쪽 차선 생성
        if lane_info.left_x == w // 2 and lane_info.right_x != w // 2:
            left_x, left_slope, left_intercept = self.generate_left_lane_from_right(
                lane_info.right_x, lane_info.right_slope, lane_info.right_intercept, w, lane_width_pixels=200
            )
            lane_info.left_x = left_x
            lane_info.left_slope = left_slope
            lane_info.left_intercept = left_intercept
            self._log(f"왼쪽 차선 미검출, 오른쪽 차선 기준으로 생성: Left X={lane_info.left_x:.1f}, Slope={lane_info.left_slope:.3f}")
        
        # 오른쪽 차선이 검출되지 않았을 때 왼쪽 차선을 기준으로 오른쪽 차선 생성
        elif lane_info.right_x == w // 2 and lane_info.left_x != w // 2:
            # 왼쪽 차선을 기준으로 오른쪽 차선 생성 (180픽셀 오른쪽)
            right_x = lane_info.left_x + 200
            right_slope = lane_info.left_slope  # 기울기는 동일
            right_intercept = lane_info.left_intercept + 200  # y절편도 동일한 간격만큼 조정
            
            # 프레임 범위 내로 제한
            right_x = max(0, min(right_x, w - 1))
            
            lane_info.right_x = right_x
            lane_info.right_slope = right_slope
            lane_info.right_intercept = right_intercept
            self._log(f"오른쪽 차선 미검출, 왼쪽 차선 기준으로 생성: Right X={lane_info.right_x:.1f}, Slope={lane_info.right_slope:.3f}")
        
        # 왼쪽 차선 직선 그리기 (slope/intercept 방식)
        if lane_info.left_slope != 0.0:
            try:
                y_bot, y_top = h, 100
                x_bot = lane_info.left_slope * y_bot + lane_info.left_intercept
                x_top = lane_info.left_slope * y_top + lane_info.left_intercept
                cv2.line(img,
                         (int(round(x_bot)), y_bot),
                         (int(round(x_top)), y_top),
                         (255, 0, 0), 2)  # 파란색
            except:
                pass
                
        # 오른쪽 차선 직선 그리기 (slope/intercept 방식)
        if lane_info.right_slope != 0.0:
            try:
                y_bot, y_top = h, 100
                x_bot = lane_info.right_slope * y_bot + lane_info.right_intercept
                x_top = lane_info.right_slope * y_top + lane_info.right_intercept
                cv2.line(img,
                         (int(round(x_bot)), y_bot),
                         (int(round(x_top)), y_top),
                         (0, 0, 255), 2)  # 빨간색
            except:
                pass
        # 기울기 등 텍스트로 표시
        cv2.putText(img, f"Left slope: {lane_info.left_slope:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(img, f"Right slope: {lane_info.right_slope:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # === 끝 ===

        # Kanayama 제어기 사용
        # 기본 조향각과 속도 계산
        base_steering_angle, calculated_speed = self.kanayama_control(lane_info)
        
        # 강건한 조향각 계산 (히스토리 적용)
        steering_angle = self.get_robust_steering_angle(lane_info, base_steering_angle)
        
        # 차선 정보 디버깅 출력
        self._log(f"Left: x={lane_info.left_x:.1f}, slope={lane_info.left_slope:.3f}")
        self._log(f"Right: x={lane_info.right_x:.1f}, slope={lane_info.right_slope:.3f}")
        self._log(f"Base steering: {base_steering_angle:.2f}°, Final: {steering_angle:.2f}°")
        self._log(f"Calculated speed: {calculated_speed:.1f} m/s")
        self._log(f"History size: {len(self.steering_history)}, No lane count: {self.no_lane_detection_count}")
        self._log("-" * 50)  # 구분선 추가
        
        # === 최종 주행각도 영상에 표시 ===
        cv2.putText(img, f"Steering Angle: {steering_angle:.2f}°", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
        cv2.putText(img, f"Speed: {calculated_speed:.1f} m/s", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # === 끝 ===

        # 바운딩 박스 정보를 BEV 좌표계로 변환 (이미 BEV 좌표계에 있음)
        bev_boxes = []
        if len(boxes) > 0:
            # 이미 BEV 좌표계에서 예측된 박스이므로 단순히 ROI 기준으로 좌표 조정
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box[1], box[0], box[3], box[2]
                
                # 256x256 리사이즈된 이미지에서 원본 BEV ROI 크기로 스케일 조정
                # 원본 BEV ROI 크기: (w, h-300) = (w, h-300)
                # 리사이즈된 크기: (256, 256)
                scale_x = original_bev.shape[1] / 256.0
                scale_y = original_bev.shape[0] / 256.0
                
                # 스케일 조정된 좌표
                bev_x1 = x1 * scale_x
                bev_y1 = y1 * scale_y
                bev_x2 = x2 * scale_x
                bev_y2 = y2 * scale_y
                
                # ROI 영역 내에 있는 박스만 포함
                if bev_y2 > 0 and bev_y1 < original_bev.shape[0]:  # ROI 영역 내
                    bev_boxes.append({
                        'x1': int(bev_x1),
                        'y1': int(bev_y1),
                        'x2': int(bev_x2),
                        'y2': int(bev_y2),
                        'class': classes[i] if i < len(classes) else 'unknown',
                        'score': scores[i] if i < len(scores) else 0.0
                    })

        # 결과를 딕셔너리로 반환
        return {
            'steering_angle': steering_angle,
            'speed': calculated_speed,
            'original': img,  # 기존 처리된 이미지
            'bev': original_bev,  # BEV 변환된 원본 영상
            'bev_boxes': bev_boxes,  # BEV 좌표계의 바운딩 박스
            'lane_info': lane_info,  # 차선 정보
            'processing_time': end_time - start_time
        }

#!/usr/bin/env python3
"""
조향각 히스토리 기능 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from image_processor import ImageProcessor, LaneInfo
from config import HISTORY_CONFIG
import numpy as np

def create_mock_dpu():
    """테스트용 Mock DPU 객체 생성"""
    class MockDPU:
        def __init__(self):
            self.initialized = True
            
        def get_input_tensors(self):
            class MockTensor:
                def __init__(self):
                    self.dims = [1, 3, 256, 256]
            return [MockTensor()]
            
        def get_output_tensors(self):
            class MockTensor:
                def __init__(self):
                    self.dims = [1, 255, 8, 8]
                    
                def get_data_size(self):
                    return 16320
                    
            return [MockTensor(), MockTensor()]
            
        def execute_async(self, input_data, output_data):
            return 0
            
        def wait(self, job_id):
            pass
    
    return MockDPU()

def test_steering_history():
    """조향각 히스토리 기능 테스트"""
    print("=== 조향각 히스토리 기능 테스트 ===")
    
    # Mock DPU와 ImageProcessor 생성
    mock_dpu = create_mock_dpu()
    processor = ImageProcessor(mock_dpu, "test_classes.txt", np.array([[10, 14], [23, 27]]))
    
    # 테스트 시나리오: 차선이 보이다가 사라지는 상황 시뮬레이션
    test_scenarios = [
        # (left_x, right_x, expected_behavior)
        (100, 150, "차선 양쪽 모두 보임 - 스무딩 적용"),
        (120, 140, "차선 양쪽 모두 보임 - 스무딩 적용"),
        (130, 130, "차선 미검출 - 히스토리 평균 사용"),
        (130, 130, "차선 미검출 - 히스토리 평균 사용"),
        (130, 130, "차선 미검출 - 히스토리 평균 사용"),
        (110, 160, "차선 양쪽 모두 보임 - 스무딩 적용"),
        (130, 130, "차선 미검출 - 히스토리 평균 사용"),
    ]
    
    print(f"설정값:")
    print(f"  - 최대 히스토리 크기: {HISTORY_CONFIG['max_history_size']}")
    print(f"  - 평균 윈도우 크기: {HISTORY_CONFIG['avg_window_size']}")
    print(f"  - 최대 차선 미검출 프레임: {HISTORY_CONFIG['max_no_lane_frames']}")
    print(f"  - 스무딩 팩터: {HISTORY_CONFIG['smoothing_factor']}")
    print()
    
    for i, (left_x, right_x, description) in enumerate(test_scenarios):
        print(f"--- 테스트 {i+1}: {description} ---")
        
        # LaneInfo 생성
        lane_info = LaneInfo()
        lane_info.left_x = left_x
        lane_info.right_x = right_x
        lane_info.left_slope = 0.1
        lane_info.right_slope = -0.1
        
        # 현재 조향각 계산 (시뮬레이션)
        if left_x != 130 and right_x != 130:
            # 차선이 보이는 경우 - 간단한 조향각 계산
            center_x = (left_x + right_x) / 2
            current_steering = (center_x - 128) * 0.5  # 간단한 비례 제어
        else:
            # 차선이 보이지 않는 경우 - 기본값
            current_steering = 0.0
        
        # 강건한 조향각 계산
        robust_steering = processor.get_robust_steering_angle(lane_info, current_steering)
        
        print(f"  차선 정보: left_x={left_x}, right_x={right_x}")
        print(f"  현재 조향각: {current_steering:.2f}°")
        print(f"  최종 조향각: {robust_steering:.2f}°")
        print(f"  히스토리 크기: {len(processor.steering_history)}")
        print(f"  차선 미검출 카운트: {processor.no_lane_detection_count}")
        
        if len(processor.steering_history) > 0:
            recent_values = list(processor.steering_history)[-3:]  # 최근 3개 값
            print(f"  최근 히스토리: {[f'{v:.2f}°' for v in recent_values]}")
        
        print()

def test_history_overflow():
    """히스토리 오버플로우 테스트"""
    print("=== 히스토리 오버플로우 테스트 ===")
    
    mock_dpu = create_mock_dpu()
    processor = ImageProcessor(mock_dpu, "test_classes.txt", np.array([[10, 14], [23, 27]]))
    
    # 히스토리 크기보다 많은 값 추가
    max_size = HISTORY_CONFIG['max_history_size']
    print(f"히스토리 최대 크기: {max_size}")
    
    for i in range(max_size + 5):
        processor.add_steering_to_history(float(i))
        print(f"  추가 {i+1}: 히스토리 크기 = {len(processor.steering_history)}")
    
    print(f"최종 히스토리 크기: {len(processor.steering_history)}")
    print(f"히스토리 내용: {list(processor.steering_history)}")
    print()

def test_average_calculation():
    """평균 계산 테스트"""
    print("=== 평균 계산 테스트 ===")
    
    mock_dpu = create_mock_dpu()
    processor = ImageProcessor(mock_dpu, "test_classes.txt", np.array([[10, 14], [23, 27]]))
    
    # 테스트 데이터 추가
    test_values = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    for val in test_values:
        processor.add_steering_to_history(val)
    
    print(f"전체 히스토리: {list(processor.steering_history)}")
    
    # 다양한 윈도우 크기로 평균 계산
    for window_size in [3, 5, 7]:
        avg = processor.get_average_steering(window_size)
        expected = sum(test_values[-window_size:]) / window_size
        print(f"  윈도우 크기 {window_size}: 계산값={avg:.2f}°, 예상값={expected:.2f}°")
    
    print()

if __name__ == "__main__":
    test_steering_history()
    test_history_overflow()
    test_average_calculation()
    
    print("=== 테스트 완료 ===") 
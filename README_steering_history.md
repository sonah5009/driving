# 조향각 히스토리 관리 기능

## 개요

양쪽 차선이 모두 보이지 않을 때 이전 조향값들을 평균내서 사용하는 기능을 구현했습니다. 이를 통해 차선 검출이 실패하는 상황에서도 안정적인 주행이 가능합니다.

## 주요 기능

### 1. 히스토리 저장
- 최근 10개의 조향각을 `deque`로 저장
- 자동으로 오래된 값 제거 (FIFO 방식)

### 2. 평균 계산
- 최근 5개 조향각의 평균 계산
- 설정 가능한 윈도우 크기

### 3. 스무딩 적용
- 현재 조향각과 이전 조향각의 가중 평균
- 급격한 조향 변화 방지

### 4. 차선 미검출 대응
- 양쪽 차선이 모두 보이지 않을 때 히스토리 사용
- 최대 5프레임까지 이전 값 사용
- 그 이후에는 기본값(직진) 사용

## 설정 파라미터

`config.py`의 `HISTORY_CONFIG`에서 다음 값들을 조정할 수 있습니다:

```python
HISTORY_CONFIG = {
    'max_history_size': 10,      # 최대 히스토리 크기
    'avg_window_size': 5,        # 평균 계산에 사용할 프레임 수
    'max_no_lane_frames': 5,     # 최대 차선 미검출 프레임 수
    'default_steering_angle': 0.0,  # 기본 조향각 (직진)
    'smoothing_factor': 0.8      # 스무딩 팩터 (0~1, 높을수록 부드러움)
}
```

## 동작 방식

### 1. 차선이 보이는 경우
```
현재 조향각 → 스무딩 적용 → 히스토리에 저장 → 최종 조향각
```

### 2. 차선이 보이지 않는 경우
```
히스토리 평균 계산 → 최종 조향각
```

### 3. 오래 차선을 못 찾는 경우
```
기본값(직진) 사용 → 최종 조향각
```

## 사용 예시

```python
# ImageProcessor 초기화
processor = ImageProcessor(dpu, classes_path, anchors)

# 프레임 처리 (자동으로 히스토리 적용)
steering_angle, processed_img = processor.process_frame(frame, use_kanayama=True)

# 수동으로 히스토리 관리
lane_info = processor.extract_lane_info(boxes, classes, img)
base_steering, speed = processor.kanayama_control(lane_info)
robust_steering = processor.get_robust_steering_angle(lane_info, base_steering)
```

## 디버깅 출력

실행 시 다음과 같은 정보가 출력됩니다:

```
Left: x=100.0, slope=0.100
Right: x=150.0, slope=-0.100
Base steering: 12.50°, Final: 10.00°
History size: 5, No lane count: 0
스무딩 적용: 12.50° → 10.00°

차선 미검출: 이전 5개 값 평균 사용 (8.50°)
```

## 테스트

테스트 스크립트를 실행하여 기능을 확인할 수 있습니다:

```bash
cd driving
python test_steering_history.py
```

## 장점

1. **안정성 향상**: 차선 검출 실패 시에도 부드러운 주행
2. **노이즈 제거**: 스무딩으로 급격한 조향 변화 방지
3. **설정 가능**: 다양한 환경에 맞게 파라미터 조정 가능
4. **하위 호환성**: 기존 코드와 호환되며 점진적 적용 가능

## 주의사항

1. **메모리 사용**: 히스토리 저장으로 약간의 메모리 사용량 증가
2. **지연 효과**: 이전 값 사용으로 인한 약간의 응답 지연
3. **설정 조정**: 환경에 맞게 파라미터 조정 필요

## 향후 개선 방안

1. **적응형 윈도우**: 속도에 따른 동적 윈도우 크기 조정
2. **가중 평균**: 최근 값에 더 높은 가중치 부여
3. **컨텍스트 인식**: 도로 상황에 따른 히스토리 사용 전략 변경 
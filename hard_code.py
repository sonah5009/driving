import time

class HardCodeController:
    def __init__(self, motor_controller, parking_config):
        self.motor_controller = motor_controller
        self.parking_config = parking_config

    def _stop_vehicle(self):
        """차량 정지"""
        self.motor_controller.reset_motor_values()
    
    def _move_forward(self):
        """전진"""
        speed = self.parking_config['forward_speed']
        print(f"[_move_forward] 전진: {speed} m/s")
        self.motor_controller.left_speed = speed
        self.motor_controller.right_speed = speed
        self.motor_controller.set_left_speed(speed)
        self.motor_controller.set_right_speed(speed)
    
    def _move_backward(self):
        """후진"""
        speed = self.parking_config['backward_speed']
        print(f"[_move_backward] 후진: {speed} m/s")
        self.motor_controller.left_speed = -speed
        self.motor_controller.right_speed = -speed
        self.motor_controller.set_left_speed(-speed)
        self.motor_controller.set_right_speed(-speed)
    
    def _turn_left(self):
        """좌회전 - 설정된 각도로 조향"""
        speed = self.parking_config['steering_speed']
        angle = self.parking_config['left_turn_angle']
        print(f"[_turn_left] 좌회전: {angle}도")
        self.motor_controller.control_motors_parking(angle, speed, 'left')
    
    def _turn_right(self):
        """우회전 - 설정된 각도로 조향"""
        speed = self.parking_config['steering_speed']
        angle = self.parking_config['right_turn_angle']
        print(f"[_turn_right] 우회전: {angle}도")
        self.motor_controller.control_motors_parking(angle, speed, 'right')
    
    def _straight_steering_from_left(self):
        """직진 조향 - 0도로 조향"""
        speed = self.parking_config['steering_speed']
        print(f"[_straight_steering] 직진: 0도 • 조향속도: {speed} m/s")
        self.motor_controller.control_motors_parking(+30, speed, 'straight_left')
    
    def _straight_steering_from_right(self):
        """직진 조향 - 0도로 조향"""
        speed = self.parking_config['steering_speed']
        print(f"[_straight_steering] 직진: 0도 • 조향속도: {speed} m/s")
        self.motor_controller.control_motors_parking(-30, speed, 'straight_right')
    

    def _straight_steering(self):
        steer_pwm = self.parking_config['steering_speed']  # 예: 30
        print("[_straight_steering] 핸들 중앙 정렬")
        # 핸들 중앙만 맞추고 차는 정지 상태
        self.motor_controller.control_motors_parking(
            angle      = 0,
            speed      = steer_pwm,
            direction  = 'center',
            settle_time= 1.0,
            coarse     = False         # 정밀 모드
        )

    
    
    def run_custom_sequence(self):
        """5초간 정방향 전진, 1초간 정지, 왼쪽 조향 후 전진 3초, 우조향 후 후진 5초 시퀀스"""
        import time
        print("[시퀀스] 3초간 바퀴 정렬 시작")
        self._straight_steering()
        time.sleep(3)
        print("[시퀀스] 20초간 정방향 전진 시작")
        self._move_forward()
        time.sleep(3)
        
        print("[시퀀스] 1초간 정지")
        self._stop_vehicle()
        time.sleep(1)
        
        print("[시퀀스] 왼쪽 조향 후 6초간 전진")
        self._turn_left()
        time.sleep(0.4)  # 조향 후 대기
        self._move_forward()
        time.sleep(5)
        
        print("[시퀀스] 오른쪽 조향 후 5초간 후진")
        self._turn_right()
        time.sleep(0.3)  # 조향 후 대기
        self._move_backward()
        time.sleep(5)
        
        print("[시퀀스] 정방향(0도) 후진 3초")
        self._straight_steering()
        time.sleep(0.4)  # 조향 후 대기
        self._move_backward()
        time.sleep(3)
        
        print("[시퀀스] 2초간 정지")
        self._stop_vehicle()
        time.sleep(2)
        
        print("[시퀀스] 2초간 정방향 전진")
        self._straight_steering()
        self._move_forward()
        time.sleep(4)
        
        print("[시퀀스] 우회전 후 3초간 전진")
        # self.parking_config["right_turn_angle"] = 
        self._turn_right()
        time.sleep(0.4)  # 조향 후 대기
        self._move_forward()
        time.sleep(5)
        
        print("[시퀀스] 좌회전 후 2초간 후진")
        self._turn_left()
        time.sleep(0.4)  # 조향 후 대기
        self._move_backward()
        time.sleep(2)
        
        print("[시퀀스] 10초간 정방향 전진")
        self._straight_steering()
        time.sleep(1)
        self._move_forward()
        time.sleep(10)
        
        print("[시퀀스] 시퀀스 완료, 차량 정지")
        self._stop_vehicle()
    
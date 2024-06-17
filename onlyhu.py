import torch
import cv2
import numpy as np
import pygame
import pyzed.sl as sl

# Pygame 초기화 및 알람 소리 설정
pygame.init()
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('C:/car/stopCHU.wav')

# YOLOv5 모델 로드
model_person = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 스테레오 카메라 초기화
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.coordinate_units = sl.UNIT.METER
zed.open(init_params)

# 알람 상태 초기화
alarm_playing = False
alarm_duration = 1000
last_alarm_time = pygame.time.get_ticks()

def get_person_distance(image, depth_map):
    results = model_person(image)
    for det in results.xyxy[0]:
        xmin, ymin, xmax, ymax, conf, cls = det
        if int(cls) == 0:  # 사람 클래스
            person_center_x = int((xmin + xmax) / 2)
            person_center_y = int((ymin + ymax) / 2)
            distance = depth_map.get_value(person_center_x, person_center_y)[1]
            return distance, (int(xmin), int(ymin), int(xmax), int(ymax)), conf
    return None, None, None

while True:
    # 스테레오 카메라 프레임 읽기
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        image = sl.Mat()
        depth_map = sl.Mat()
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

        image_np = image.get_data()[:, :, :3]

        # 사람 인식 및 거리 측정
        person_distance, person_bbox, person_conf = get_person_distance(image_np, depth_map)

        # 사람이 10미터 이내에 있을 경우 알람 재생
        if person_distance is not None and person_distance < 3:
            current_time = pygame.time.get_ticks()
            if not alarm_playing or (current_time - last_alarm_time > alarm_duration):
                alarm_sound.play()
                last_alarm_time = current_time
                alarm_playing = True

        # 시각화
 
        cv2.imshow('Stereo Camera - Person Detection', image_np)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# 리소스 해제
zed.close()
cv2.destroyAllWindows()
pygame.mixer.quit()

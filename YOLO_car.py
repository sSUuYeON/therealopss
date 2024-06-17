import torch
import cv2
import numpy as np
import pygame
import pyzed.sl as sl

# Pygame 초기화 및 알람 소리 설정
pygame.init()
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('C:/car/stopCHU.wav')  # 알람 소리 파일 경로

# YOLOv5 모델 로드 (사전 학습된 모델)
model_car = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model_person = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 일반 웹캡 캡처 초기화 (사람 인식용)
cap_person = cv2.VideoCapture(0)

# 스테레오 카메라 초기화 (자동차 인식 및 거리 측정용)
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Depth 모드 설정
init_params.coordinate_units = sl.UNIT.METER  # 거리 단위 설정
zed.open(init_params)

# 프레임 카운터 초기화
frame_counter = 0
alarm_playing = False
alarm_duration = 1000  # 알람 소리 지속 시간 (밀리초)
last_alarm_time = pygame.time.get_ticks()

def get_car_distance(image, depth_map):
    results = model_car(image)
    for det in results.xyxy[0]:
        xmin, ymin, xmax, ymax, conf, cls = det
        if int(cls) == 2:  # 자동차 클래스
            car_center_x = int((xmin + xmax) / 2)
            car_center_y = int((ymin + ymax) / 2)
            distance = depth_map.get_value(car_center_x, car_center_y)[1]
            return distance, (int(xmin), int(ymin), int(xmax), int(ymax)), conf
    return None, None, None

while True:
    # 사람 인식용 프레임 읽기
    ret_person, frame_person = cap_person.read()
    if not ret_person:
        break

    # 스테레오 카메라 프레임 읽기
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        image = sl.Mat()
        depth_map = sl.Mat()
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

        image_np = image.get_data()[:, :, :3]  # RGB 이미지 변환

        # 자동차 인식 및 거리 측정
        car_distance, car_bbox, car_conf = get_car_distance(image_np, depth_map)

        # 사람 인식
        results_person = model_person(frame_person)
        person_detected = any(int(cls) == 0 for *_, cls in results_person.xyxy[0])

        # 자동차가 10미터 이내에 있고 사람이 인식되었을 경우 알람 재생
        if car_distance is not None and car_distance < 10 and person_detected:
            current_time = pygame.time.get_ticks()
            if not alarm_playing or (current_time - last_alarm_time > alarm_duration):
                alarm_sound.play()
                last_alarm_time = current_time
                alarm_playing = True

        # 시각화
        if car_bbox:
            xmin, ymin, xmax, ymax = car_bbox
            cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image_np, f'car {car_conf:.2f} {car_distance:.2f}m', (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Stereo Camera - Car Detection', image_np)

    # 사람 인식 시각화
    for det in results_person.xyxy[0]:
        xmin, ymin, xmax, ymax, conf, cls = det
        if int(cls) == 0:  # 사람 클래스
            cv2.rectangle(frame_person, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
            cv2.putText(frame_person, f'person {conf:.2f}', (int(xmin), int(ymin) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Webcam - Person Detection', frame_person)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# 리소스 해제
cap_person.release()
zed.close()
cv2.destroyAllWindows()
pygame.mixer.quit()

# 어린이 무단 횡단 방지 프로젝트
어린이 보호 구역의 횡단보도에서 무단횡단을 실시간으로 감지하고 경고를 제공하는 시스템 개발

무단횡단을 줄이고 어린이들의 교통 안전을 강화

yolo5 : 사람 인식

pyzed : 거리 인식

---

# 실행 방법:

1. onlyhu.py, stopCHU.wav, yolov5s.pt 파일을 다운로드
 
2. c드라이브에 빈 폴더를 생성 후 이름을 car로 변경
 
3. car 폴더로 다운로드 받은 세 파일을 이동
 
4. 필요 모듈을 다운받고 실행

---
# 파이썬, 모듈 버전 세팅, 순서

: python==3.8.19

numpy==1.23.5

requests==2.32.3

pyzed==4.1

cython==3.0.10

pandas==2.0.3

pillow==10.3.0

pygame==2.5.2

torch==2.3.1

---

# pyzed 설치 방법

https://www.stereolabs.com/developers/release

에 접속 후 본인 컴퓨터에 설치 되어있는 cuda 버전에 맞는 zed sdk 다운로드(cuda 설치가 선행되어야함)

zed sdk파일에서 get_python_api.py를 c드라이브 폴더로 이동

get_python__api.py 실행을 위해 requests 모듈 필요. 설치 후 실행하면 pyzed가 설치됨.

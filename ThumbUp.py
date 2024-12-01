import cv2
import mediapipe as mp
import time
import numpy as np
from PIL import Image, ImageDraw

# MediaPipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# 엄지 손가락이 위로 향한 경우 감지 함수
def detect_thumb_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    
    # 엄지 손가락이 위로 향한 경우 (Y값 비교)
    if thumb_tip.y < thumb_ip.y:
        return True
    return False

# 하트 그리기 함수
def draw_heart(image):
    # Pillow로 이미지를 처리하기 위한 부분
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    # 하트 모양을 그릴 좌표 (기본적인 하트 형태)
    heart_points = [
        (150, 200), (100, 150), (50, 100), (50, 50),
        (100, 30), (150, 50), (200, 30), (250, 50),
        (250, 100), (200, 150), (150, 200)
    ]
    
    # 하트 모양을 그리기 (빨간색)
    draw.polygon(heart_points, fill="red")

    # 변환된 이미지를 다시 numpy 배열로 변환
    image = np.array(pil_image)
    return image

# 카메라 캡처 시작
cap = cv2.VideoCapture(0)
thumb_up_start_time = None  # 엄지척 시작 시간
thumb_up_duration = 3  # 3초 이상 유지되면 동작
heart_display_start_time = None  # 하트가 표시된 시간
heart_display_duration = 10  # 하트가 10초 동안 표시되도록 설정
heart_shown = False  # 하트가 이미 표시되었는지 확인하는 변수

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라를 찾을 수 없습니다.")
            continue

        # BGR 이미지를 RGB로 변환
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # RGB 이미지를 다시 BGR로 변환
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        thumb_up_detected = False  # 엄지척이 감지되었는지 여부

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 엄지 손가락이 위로 향하면 하트 그리기
                if detect_thumb_up(hand_landmarks):
                    if thumb_up_start_time is None:
                        thumb_up_start_time = time.time()  # 엄지척 시작 시간 기록
                    thumb_up_detected = True
                else:
                    thumb_up_start_time = None  # 엄지척이 아닌 경우 타이머 초기화

                # 손 랜드마크 시각화
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # 엄지척이 3초 이상 유지되면 하트 그리기
        if thumb_up_detected and thumb_up_start_time and (time.time() - thumb_up_start_time >= thumb_up_duration):
            print("엄지척 감지!")  # 터미널에 "엄지척 감지!" 출력
            if not heart_shown:
                image = draw_heart(image)  # 하트 그리기 함수 호출
                heart_display_start_time = time.time()  # 하트 표시 시작 시간 기록
                heart_shown = True  # 하트가 표시되었음을 표시
            thumb_up_start_time = None  # 3초 이후에는 타이머 리셋

        # 하트가 표시된 후 10초가 지나면 화면에서 하트 제거
        if heart_shown and (time.time() - heart_display_start_time >= heart_display_duration):
            heart_shown = False  # 하트 표시 완료
            heart_display_start_time = None  # 하트 표시 타이머 리셋

        # 화면에 손 랜드마크 표시
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:  # ESC 키를 누르면 종료
            break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()

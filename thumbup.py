import cv2
import mediapipe as mp
import pyautogui
import random
import time

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

# "짱♡" 출력 함수
def display_text(image, text="짱♡", duration=15):
    start_time = time.time()
    while time.time() - start_time < duration:
        # 화면에 "짱♡" 출력
        pyautogui.write(text, interval=0.1)
        # 화면의 랜덤 위치에 이동
        x = random.randint(100, 500)
        y = random.randint(100, 500)
        pyautogui.moveTo(x, y)  # 텍스트를 랜덤 위치로 이동

# 카메라 캡처 시작
cap = cv2.VideoCapture(0)
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

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 엄지 손가락이 위로 향하면 "짱♡" 출력
                if detect_thumb_up(hand_landmarks):
                    print("Thumb up detected! 짱♡")
                    display_text(image)  # "짱♡" 출력 함수 호출

                # 손 랜드마크 시각화
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # 화면에 손 랜드마크 표시
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:  # ESC 키를 누르면 종료
            break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()

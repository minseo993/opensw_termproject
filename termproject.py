import cv2
import mediapipe as mp
import time  # 사진 저장 시 타임스탬프를 사용

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def is_fist(landmarks):
    """
    주먹을 쥔 상태인지 확인하는 함수.
    손가락 끝 랜드마크가 손목 기준 아래에 있는지를 확인.
    """
    # 손가락 끝 랜드마크 (4, 8, 12, 16, 20)
    tip_ids = [4, 8, 12, 16, 20]
    
    # 손목 랜드마크 y값
    wrist_y = landmarks[0].y
    
    # 손가락 끝들이 손목 아래 있는지 체크
    for tip_id in tip_ids:
        if landmarks[tip_id].y < wrist_y:  # 손가락 끝이 손목 위에 있다면 주먹이 아님
            return False
    return True


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

    # 이미지 전처리
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # 이미지 후처리
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # 주먹 감지
        if is_fist(hand_landmarks.landmark):
            print("주먹 감지! 사진을 저장합니다.")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"fist_detected_{timestamp}.jpg"
            cv2.imwrite(filename, image)
            print(f"사진 저장 완료: {filename}")

    # 화면 출력
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:  # ESC 키로 종료
      break

cap.release()
cv2.destroyAllWindows()

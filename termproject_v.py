import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# V 제스처를 인식하는 함수
def is_v_sign(hand_landmarks):
    # 손 랜드마크 인덱스
    INDEX_FINGER_TIP = 8
    INDEX_FINGER_MCP = 5
    MIDDLE_FINGER_TIP = 12
    MIDDLE_FINGER_MCP = 9
    RING_FINGER_TIP = 16
    RING_FINGER_MCP = 13
    PINKY_TIP = 20
    PINKY_MCP = 17
    THUMB_TIP = 4
    THUMB_IP = 3

    # 손가락의 끝과 뿌리 사이 거리 계산
    def distance(pt1, pt2):
        return np.linalg.norm(
            np.array([pt1.x, pt1.y]) - np.array([pt2.x, pt2.y])
        )
    
    # 각 손가락이 펴졌는지 접혔는지 확인
    index_extended = distance(hand_landmarks.landmark[INDEX_FINGER_TIP],
                              hand_landmarks.landmark[INDEX_FINGER_MCP]) > 0.1
    middle_extended = distance(hand_landmarks.landmark[MIDDLE_FINGER_TIP],
                                hand_landmarks.landmark[MIDDLE_FINGER_MCP]) > 0.1
    ring_folded = distance(hand_landmarks.landmark[RING_FINGER_TIP],
                            hand_landmarks.landmark[RING_FINGER_MCP]) < 0.05
    pinky_folded = distance(hand_landmarks.landmark[PINKY_TIP],
                            hand_landmarks.landmark[PINKY_MCP]) < 0.05
    thumb_folded = distance(hand_landmarks.landmark[THUMB_TIP],
                             hand_landmarks.landmark[THUMB_IP]) < 0.05

    # V 제스처 조건
    return index_extended and middle_extended and ring_folded and pinky_folded and thumb_folded

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    flash_active = False
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라를 찾을 수 없습니다.")
            continue

        # 이미지를 RGB로 변환
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # 이미지를 다시 BGR로 변환
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 랜드마크를 그리고 브이 제스처 확인
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                if is_v_sign(hand_landmarks):
                    flash_active = True

        # 플래시 효과
        if flash_active:
            flash_active = False
            flash_frame = np.ones_like(image) * 255  # 하얀 화면
            cv2.imshow('MediaPipe Hands', flash_frame)
            cv2.waitKey(100)  # 100ms 동안 플래시 유지

        # 화면 표시
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

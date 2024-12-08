import cv2
import mediapipe as mp
import random
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 손하트를 만들었는지 확인하는 함수
def is_hand_heart(landmarks):
    # 엄지와 검지가 서로 가까운지 확인 (일정 간격 이하로 가까운지)
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    
    # 엄지와 검지의 간격 (x, y 좌표의 거리)
    thumb_index_distance = abs(thumb_tip.x - index_tip.x) + abs(thumb_tip.y - index_tip.y)
    
    # 하트 모양이 되었는지
    heart_thumb = landmarks[2].y <= landmarks[3].y <= landmarks[4].y
    heart_index = landmarks[7].y <= landmarks[8].y <= landmarks[9].y
    
    # 두 손이 손하트를 만들었을 때 특정 거리 이하로 가까우면 하트로 판단
    return thumb_index_distance < 1  # 일정 범위 내로 가까운 경우

# 손가락들이 올바르게 펼쳐졌는지 확인하는 함수 (손하트를 위한 조건)
def are_fingers_raised_for_heart(landmarks):
    # 나머지 손가락들이 펼쳐져 있는지 확인
    middle_up = landmarks[12].y < landmarks[10].y  # 중지
    ring_up = landmarks[16].y < landmarks[14].y  # 약지
    pinky_up = landmarks[20].y < landmarks[18].y  # 새끼
    
    # 엄지는 하트 모양에 포함되므로 펼쳐져 있지 않아야 한다
    thumb_up = landmarks[4].y <= landmarks[3].y  # 엄지
    index_up = landmarks[8].y <= landmarks[6].y  # 검지
    
    # 나머지 손가락들이 모두 펼쳐져 있어야 하고, 엄지는 접혀 있어야 한다
    return middle_up and ring_up and pinky_up and not thumb_up and not index_up

# "!LOVE!" 텍스트를 화면의 중앙에 출력하는 함수
def draw_love_text(image, landmarks):
    height, width, _ = image.shape

    # 랜덤한 색상 생성 (RGB)
    color = (random.randint(180, 255), random.randint(0, 100), random.randint(0,100))

    # 텍스트 위치를 손목 위치에 맞추기 (예: 손목 위치를 기준으로)
    # landmarks[9] (검지 끝)와 landmarks[4] (엄지 끝)의 좌표를 사용하여 텍스트 위치 계산
    x = int((landmarks[9].x + landmarks[4].x) * width / 2)
    y = int((landmarks[9].y + landmarks[4].y) * height / 2)

    # 텍스트 크기 계산
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 4
    thickness = 15
    text_size = cv2.getTextSize("LOVE", font, font_scale, thickness)[0]

    # 텍스트 크기를 기준으로 중심이 (x, y)로 오도록 좌표 조정
    x -= text_size[0] // 2
    y += text_size[1] // 2  # 텍스트 중심을 맞추기 위해 y 위치를 아래로 조정

    return (x, y), color  # 텍스트 출력 위치와 색상 반환

cap = cv2.VideoCapture(0)
running = True  # 프로그램 실행 상태
last_love_time = 0  # "!LOVE!" 텍스트를 출력한 시간 기록
love_position = None  # "!LOVE!" 텍스트의 마지막 위치
love_color = None  # "!LOVE!" 텍스트의 색상

font = cv2.FONT_HERSHEY_SIMPLEX  # 기본 폰트 설정
font_scale = 4  # 폰트 크기
thickness = 15  # 폰트 두께

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened() and running:
        success, image = cap.read()
        if not success:
            print("can not use camera")
            continue

        image_height, image_width, _ = image.shape
        image.flags.writeable = False  # 이미지 쓰기 방지 모드로 변경 (성능 최적화)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 이미지를 RGB로 변환
        results = hands.process(image)  # Mediapipe로 손 탐지 수행

        image.flags.writeable = True  # 다시 쓰기 가능 모드로 변경
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 이미지를 다시 BGR로 변환 (OpenCV용)

        hands_with_heart = 0  # 손하트를 만든 손의 개수

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 손의 랜드마크를 그림
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

                # 손하트를 만들었는지 확인 (엄지와 검지가 붙고 나머지 손가락들이 펼쳐짐)
                if is_hand_heart(hand_landmarks.landmark) and are_fingers_raised_for_heart(hand_landmarks.landmark):
                    hands_with_heart += 1  # 손하트를 만든 손 증가

            # 두 손이 모두 손하트를 만들었을 때만 "LOVE" 출력
            if hands_with_heart == 2:
                current_time = time.time()

                # 마지막 "LOVE"가 출력된 후 1초가 지나면 새 위치에 "LOVE"를 출력
                if current_time - last_love_time > 1:
                    love_position, love_color = draw_love_text(image, hand_landmarks.landmark)  # 손목 위치를 기준으로 텍스트 출력
                    last_love_time = current_time  # 텍스트 출력 시간 업데이트
            else:
                love_position = None  # 두 손 모두 손하트를 만들지 않으면 텍스트를 지움

        # "LOVE" 텍스트가 한 위치에 1초 동안 머물도록 하며, 그 후에 새로 나타남
        if love_position:
            x, y = love_position
            # 텍스트 그리기
            cv2.putText(image, "LOVE", (x, y), font, font_scale, love_color, thickness)

        # 결과 이미지를 화면에 출력
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC 키를 누르면 종료
            break

cap.release()  # 카메라 자원 해제
cv2.destroyAllWindows()  # 모든 창 닫기
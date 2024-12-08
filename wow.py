import cv2
import mediapipe as mp
import random
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# 손가락이 올바르게 펼쳐졌는지 확인하는 함수
def are_fingers_raised(landmarks):
    # 각 손가락이 펼쳐졌는지 확인 (손끝이 두 번째 관절 위에 있을 때)
    index_up = landmarks[8].y < landmarks[6].y  # 검지
    middle_up = landmarks[12].y < landmarks[10].y  # 중지
    ring_up = landmarks[16].y < landmarks[14].y  # 약지
    thumb_up = landmarks[4].y > landmarks[3].y  # 엄지
    pinky_up = landmarks[20].y < landmarks[19].y  # 새끼
    
    # 세 손가락만 펼쳐졌을 때만 True 반환 (새끼손가락은 펼쳐지지 않아야 함)
    return index_up and middle_up and ring_up and not thumb_up and not pinky_up

# "!WOW!" 텍스트를 다양한 색상으로 출력하고 겹치지 않게 하는 함수
def draw_wow_text(image, used_positions):
    height, width, _ = image.shape

    # 랜덤한 위치 계산 (겹침을 방지하기 위한 처리)
    while True:
        x = random.randint(0, width - 100)  # x 좌표 (최대 크기 고려)
        y = random.randint(50, height - 50)  # y 좌표 (텍스트가 화면 밖으로 나가지 않도록 조정)
        
        # 겹치지 않도록 위치가 사용되지 않은 경우에만 탈출
        if (x // 50, y // 50) not in used_positions:
            used_positions.add((x // 50, y // 50))  # 50픽셀 단위로 겹침 방지
            break

    # 랜덤한 색상 생성
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    return (x, y), color  # 텍스트 출력 위치와 색상 반환

# 텍스트가 화면 밖으로 나가지 않도록 하는 함수
def fit_text_within_bounds(image, text, position, font, font_scale, color, thickness):
    # 텍스트 크기 계산
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position

    # 텍스트가 화면 밖으로 나가지 않도록 위치 조정
    if x + text_width > image.shape[1]:  # x가 이미지의 가로 크기를 넘으면
        x = image.shape[1] - text_width  # 텍스트의 x 좌표를 오른쪽 끝으로 조정
    if y + text_height > image.shape[0]:  # y가 이미지의 세로 크기를 넘으면
        y = image.shape[0] - text_height  # 텍스트의 y 좌표를 아래쪽 끝으로 조정
    if x < 0:  # x가 0보다 작으면
        x = 0  # 텍스트의 x 좌표를 0으로 조정
    if y < 0:  # y가 0보다 작으면
        y = 0  # 텍스트의 y 좌표를 0으로 조정

    # 텍스트 그리기
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness)
    return (x, y)  # 텍스트가 그려진 새로운 위치 반환

cap = cv2.VideoCapture(0)
running = True  # 프로그램 실행 상태
used_positions = set()  # 이미 사용한 위치를 저장하는 집합
last_wow_time = 0  # "!WOW!" 텍스트를 출력한 시간 기록
wow_position = None  # "!WOW!" 텍스트의 마지막 위치
wow_color = None  # "!WOW!" 텍스트의 색상

font = cv2.FONT_HERSHEY_SIMPLEX  # 기본 폰트 설정
font_scale = 10 # 폰트 크기
thickness = 32  # 폰트 두께

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

        hands_with_three_fingers_up = 0  # 세 손가락을 펼친 손의 개수

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 손의 랜드마크를 그림
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # 손가락이 올바르게 펼쳐졌는지 확인 (검지, 중지, 약지만 펼쳐짐)
                if are_fingers_raised(hand_landmarks.landmark):
                    hands_with_three_fingers_up += 1  # 세 손가락이 펼쳐진 손 증가

            # 두 손이 모두 세 손가락을 펼쳤을 때만 "!WOW!" 출력
            if hands_with_three_fingers_up == 2:
                current_time = time.time()

                # 마지막 "!WOW!"가 출력된 후 1초가 지나면 새 위치에 "!WOW!"를 출력
                if current_time - last_wow_time > 1:
                    wow_position, wow_color = draw_wow_text(image, used_positions)  # 텍스트 출력 위치와 색상 반환
                    last_wow_time = current_time  # 텍스트 출력 시간 업데이트
            else:
                wow_position = None  # 두 손 모두 세 손가락이 펼쳐지지 않으면 텍스트를 지움

        # "!WOW!" 텍스트가 한 위치에 1초 동안 머물도록 하며, 그 후에 새로 나타남
        if wow_position:
            x, y = wow_position
            # 텍스트가 화면 밖으로 나가지 않도록 조정
            wow_position = fit_text_within_bounds(image, "!WOW!", (x, y), font, font_scale, wow_color, thickness)

        # 결과 이미지를 화면에 출력
        cv2.imshow('WOW ~ amazing!', image)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC 키를 누르면 종료
            break

cap.release()  # 카메라 자원 해제
cv2.destroyAllWindows()  # 모든 창 닫기
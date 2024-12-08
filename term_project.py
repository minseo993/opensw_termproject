import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Function to detect if the hand is open
def is_hand_open(landmarks):
    fingers_up = 0
    for i, (tip, pip) in enumerate([(8, 6), (12, 10), (16, 14), (20, 18)]):
        if landmarks[tip].y < landmarks[pip].y:  # If the fingertip is above the joint
            fingers_up += 1
    return fingers_up == 4  # Check if all 4 fingers are open

# Function to detect the "V sign" gesture
def is_v_sign(landmarks):
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_down = landmarks[16].y > landmarks[14].y
    pinky_down = landmarks[20].y > landmarks[18].y
    return index_up and middle_up and ring_down and pinky_down

# Function to detect if only the index finger is raised (program exit gesture)
def is_index_finger_only(landmarks):
    index_up = landmarks[8].y < landmarks[6].y  # Index finger is raised
    other_fingers_down = (
        landmarks[12].y > landmarks[10].y and  # Middle finger
        landmarks[16].y > landmarks[14].y and  # Ring finger
        landmarks[20].y > landmarks[18].y      # Pinky finger
    )
    return index_up and other_fingers_down

# Function to handle gestures
def handle_gestures(landmarks, image):
    global running  # Controls the program's running state
    # Open hand: Change background color to white
    if is_hand_open(landmarks):
        print("Open hand detected - Changing background color to white")
        image[:] = (255, 255, 255)  # White background

    # V sign: Change background color to green
    elif is_v_sign(landmarks):
        print("V sign detected - Changing background color to green")
        image[:] = (0, 255, 0)  # Green background

    # Index finger only: Exit program
    elif is_index_finger_only(landmarks):
        print("Index finger only detected - Exiting program")
        running = False  # Set program exit flag

cap = cv2.VideoCapture(0)
running = True  # Program running state
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened() and running:
        success, image = cap.read()
        if not success:
            print("Unable to access the camera.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

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

                landmarks = hand_landmarks.landmark
                handle_gestures(landmarks, image)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()

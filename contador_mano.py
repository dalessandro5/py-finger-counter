import cv2
import mediapipe as mp

# Inicializar MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Configurar cámara
cap = cv2.VideoCapture(0)

# IDs de las puntas de los dedos
fingerTipsIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    fingers = []

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]  # Solo una mano
        lmList = []
        for id, lm in enumerate(handLms.landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append((cx, cy))

        # Pulgar (izquierda vs derecha según orientación)
        if lmList[4][0] < lmList[3][0]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Otros dedos
        for tipId in fingerTipsIds[1:]:
            if lmList[tipId][1] < lmList[tipId - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)

        # Mostrar en pantalla
        cv2.putText(img, f'Dedos: {totalFingers}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Contador de dedos", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

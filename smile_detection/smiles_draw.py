import cv2

# Carregar os classificadores Haar Cascade pré-treinados
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Função para desenhar um rosto sorridente
def draw_smiley_face(frame, x, y, w, h, happy=True):
    # Centro do rosto ao lado do rosto detectado
    center_x = x + w + 60
    center_y = y + h // 2
    radius = w // 4

    # Desenhar a cabeça
    cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), -1)

    # Desenhar os olhos
    eye_x = center_x - radius // 2
    eye_y = center_y - radius // 4
    cv2.circle(frame, (eye_x, eye_y), radius // 8, (0, 0, 0), -1)
    cv2.circle(frame, (eye_x + radius, eye_y), radius // 8, (0, 0, 0), -1)

    # Desenhar a boca
    mouth_y = center_y + radius // 4
    if happy:
        cv2.ellipse(frame, (center_x, mouth_y), (radius // 2, radius // 4), 0, 0, 180, (0, 0, 0), -1)
    else:
        cv2.ellipse(frame, (center_x, mouth_y + radius // 4), (radius // 2, radius // 4), 0, 0, -180, (0, 0, 0), -1)

# Iniciar a captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(150, 150))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detecção de sorriso
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=25)

        if len(smiles) > 0:
            cv2.putText(frame, "Seja bem vindo", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            draw_smiley_face(frame, x, y, w, h, happy=True)
        else:
            cv2.putText(frame, "Sorria para entrar", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            draw_smiley_face(frame, x, y, w, h, happy=False)

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()

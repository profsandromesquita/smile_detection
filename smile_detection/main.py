import cv2

# Carregar os classificadores Haar Cascade pré-treinados
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Iniciar a captura de vídeo
cap = cv2.VideoCapture(0)

# Estado da detecção anterior
last_detection = None
detection_stability_count = 0  # Contador para a estabilidade da detecção

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Ajuste os parâmetros conforme necessário
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(150, 150))

    for (x, y, w, h) in faces:
        # Lógica para verificar a estabilidade da detecção
        if last_detection:
            last_x, last_y, last_w, last_h = last_detection
            if abs(x - last_x) < 10 and abs(y - last_y) < 10 and abs(w - last_w) < 10 and abs(h - last_h) < 10:
                detection_stability_count += 1
            else:
                detection_stability_count = 0
        else:
            detection_stability_count += 1

        # Atualizar a detecção anterior
        last_detection = (x, y, w, h)

        if detection_stability_count > 2:  # Verifique se a detecção é estável
            # A detecção é considerada estável
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Detecção de sorriso
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=26)

            if len(smiles) > 0:
                cv2.putText(frame, "Seja bem vindo", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Sorria para entrar", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Mostrar o resultado
    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()

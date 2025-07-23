import cv2
import mediapipe as mp
import numpy as np
import random

# Инициализация моделей
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, refine_landmarks=True)

cap = cv2.VideoCapture(0)

# Константы
HUD_FACE_SIZE = 200
HUD_INFO_WIDTH = 250
FOCAL_LENGTH = 1000
AVG_FACE_WIDTH = 0.15
UPDATE_FREQUENCY = 10  # Частота обновления бинарных цифр (каждый 10-й кадр)

def create_transparent_overlay(image, overlay, x, y):
    h, w = overlay.shape[:2]
    if x + w > image.shape[1] or y + h > image.shape[0]:
        return
    
    roi = image[y:y+h, x:x+w]
    if overlay.shape[2] == 4:
        overlay_img = overlay[:, :, :3]
        overlay_mask = overlay[:, :, 3:] / 255.0
    else:
        overlay_img = overlay
        overlay_mask = np.ones(overlay.shape[:2] + (1,))
    
    roi[:] = roi * (1 - overlay_mask) + overlay_img * overlay_mask

def get_normalized_face(landmarks, size):
    points = np.array([(lm.x, lm.y) for lm in landmarks])
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    
    if (max_coords - min_coords).max() == 0:
        return np.zeros_like(points, dtype=int)
    
    scale = size / (max_coords - min_coords).max() * 0.8
    centered = (points - min_coords) * scale
    centered += (size - centered.max(axis=0)) / 2
    
    return centered.astype(int)

def generate_binary_code(rows=4, cols=6):
    binary_code = np.zeros((rows * 20, cols * 15, 4), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            digit = random.choice(["0", "1"])
            cv2.putText(binary_code, digit, 
                       (j * 15 + 5, i * 20 + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0, 255), 1)
    return binary_code

try:
    frame_counter = 0
    binary_hud = generate_binary_code()  # Инициализация бинарного кода
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Терминатор-фильтр
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = 0
        hsv[:, :, 1] = 255
        terminator_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        terminator_frame = cv2.addWeighted(terminator_frame, 0.7, frame, 0.3, 0)

        # Детекция лиц
        face_detection_results = face_detection.process(rgb_frame)
        face_mesh_results = face_mesh.process(rgb_frame)
        
        num_faces = 0
        face_coords = []

        if face_detection_results.detections:
            num_faces = len(face_detection_results.detections)
            
            for detection in face_detection_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                cv2.rectangle(terminator_frame, 
                            (x, y), 
                            (x + width, y + height),
                            (0, 255, 0), 2)
                
                face_coords.append((x + width//2, y + height//2))

                # Добавляем "TARGET" на зелёном прямоугольнике
                target_hud = np.zeros((30, 80, 3), dtype=np.uint8)
                target_hud[:, :] = (0, 70, 0)  # Тёмно-зелёный фон
                cv2.putText(target_hud, "TARGET", (5, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                terminator_frame[y - 40:y - 10, x + width//2 - 40:x + width//2 + 40] = target_hud

        # Окно FaceMesh (левый верх)
        if face_mesh_results.multi_face_landmarks:
            face_hud = np.zeros((HUD_FACE_SIZE, HUD_FACE_SIZE, 4), dtype=np.uint8)
            try:
                normalized_face = get_normalized_face(face_mesh_results.multi_face_landmarks[0].landmark, HUD_FACE_SIZE)
                for x, y in normalized_face:
                    if 0 <= x < HUD_FACE_SIZE and 0 <= y < HUD_FACE_SIZE:
                        cv2.circle(face_hud, (x, y), 1, (0, 255, 0, 255), -1)
            except:
                pass
            
            create_transparent_overlay(terminator_frame, face_hud, 10, 10)

        # Счетчик людей (левый верх, под FaceMesh)
        counter_hud = np.zeros((60, 120, 4), dtype=np.uint8)
        cv2.putText(counter_hud, f"Humans: {num_faces}", 
                   (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0, 255), 1)
        create_transparent_overlay(terminator_frame, counter_hud, 10, HUD_FACE_SIZE + 20)

        # Случайные бинарные цифры (правый верх) - обновляем реже
        if frame_counter % UPDATE_FREQUENCY == 0:
            binary_hud = generate_binary_code()
        create_transparent_overlay(terminator_frame, binary_hud, w - 100, 10)

        # Cyberdyne Systems (левый низ)
        cyberdyne_hud = np.zeros((30, 200, 4), dtype=np.uint8)
        cv2.putText(cyberdyne_hud, "Cyberdyne Systems", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0, 255), 1)
        create_transparent_overlay(terminator_frame, cyberdyne_hud, 10, h - 40)

        # Координаты (внизу)
        if face_coords:
            coord_text = " | ".join([f"X:{x} Y:{y}" for x, y in face_coords])
            cv2.rectangle(terminator_frame, 
                         (0, h - 30), 
                         (w, h), 
                         (0, 0, 0), -1)
            cv2.putText(terminator_frame, coord_text, 
                       (10, h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Terminator Vision", terminator_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
from retinaface import RetinaFace
import cv2

model = RetinaFace

cap = cv2.VideoCapture(0)
draw_landmarks = False

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    faces = model.detect_faces(frame)

    if faces is not None:
        for key in faces.keys():
            identity = faces[key]
            facial_area = identity["facial_area"]
            cv2.rectangle(frame, (facial_area[0], facial_area[1]), (facial_area[2], facial_area[3]), (0, 255, 0), 2)
            
            if draw_landmarks:
                landmarks = identity['landmarks']
                for point in landmarks.values():
                    cv2.circle(frame, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)
    
    cv2.imshow("Face Detection with Landmarks", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('l'):
        draw_landmarks = not draw_landmarks

cap.release()
cv2.destroyAllWindows()

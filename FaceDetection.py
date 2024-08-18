from retinaface import RetinaFace
import cv2


model = RetinaFace 

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    
    if not ret:
        break
    
    faces=model.detect_faces(frame)

    if faces is not None:
        for key in faces.keys():
            identity=faces[key]
            landmarks=identity['landmarks']

            

            facial_area = identity["facial_area"]
            cv2.rectangle(frame, (facial_area[0], facial_area[1]), (facial_area[2], facial_area[3]), (0, 255, 0), 2)
    cv2.imshow("Smile Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
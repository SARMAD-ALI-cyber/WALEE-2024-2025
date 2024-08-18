import cv2
from retinaface import RetinaFace
import numpy as np


def is_smiling(landmarks):
    mouth_right = landmarks['mouth_right']
    mouth_left = landmarks['mouth_left']
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']

    # Calculate mouth width
    mouth_width = np.linalg.norm(np.array(mouth_right) - np.array(mouth_left))
    
    # Calculate eye width (as a reference for face width)
    face_width = np.linalg.norm(np.array(right_eye) - np.array(left_eye))

    # Determine if smiling 
    return (mouth_width / face_width) > 0.87


mustache_img = cv2.imread('mostache.png', cv2.IMREAD_UNCHANGED)

def apply_mustache(frame, landmarks):
    # Extract the landmarks we need
    nose = landmarks['nose']
    mouth_right = landmarks['mouth_right']
    mouth_left = landmarks['mouth_left']
    
    # Calculate the width of the mustache based on the mouth width
    mouth_width = int(np.linalg.norm(np.array(mouth_right) - np.array(mouth_left)))
    
    # Calculate the height based on the mustache image's aspect ratio
    aspect_ratio = mustache_img.shape[1] / mustache_img.shape[0]
    mustache_width = int(mouth_width * 1.2)  # Make it slightly wider than the mouth
    mustache_height = int(mustache_width / aspect_ratio)  # Maintain aspect ratio

    # Resize the mustache image
    resized_mustache = cv2.resize(mustache_img, (mustache_width, mustache_height))

    # Calculate the top-left corner for placing the mustache
    # Position it so that it sits just below the nose
    top_left_x = int(nose[0] - mustache_width / 2)
    top_left_y = int(nose[1] + (mouth_left[1] - nose[1]) * 0.3)

    # Ensure the coordinates are within the frame boundaries
    top_left_x = max(top_left_x, 0)
    top_left_y = max(top_left_y, 0)

    # Determine the region of interest (ROI) in the frame where the mustache will be placed
    roi = frame[top_left_y:top_left_y + mustache_height, top_left_x:top_left_x + mustache_width]

    # Split the channels of the mustache image
    mustache_rgb = resized_mustache[:, :, :3]
    alpha = resized_mustache[:, :, 3] / 255.0

    # Blend the mustache with the frame using the alpha channel
    for c in range(0, 3):
        roi[:, :, c] = (1.0 - alpha) * roi[:, :, c] + alpha * mustache_rgb[:, :, c]

    # Replace the region of interest with the blended result
    frame[top_left_y:top_left_y + mustache_height, top_left_x:top_left_x + mustache_width] = roi

    return frame



def apply_smile_filter(frame, landmarks):
    if is_smiling(landmarks):
        # Apply filter or overlay here
        frame = apply_mustache(frame, landmarks)  # Example: Applying glasses
        print("Smiling detected! Filter applied.")
    return frame





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

            frame=apply_smile_filter(frame, landmarks)

            facial_area = identity["facial_area"]
            cv2.rectangle(frame, (facial_area[0], facial_area[1]), (facial_area[2], facial_area[3]), (0, 255, 0), 2)
    cv2.imshow("Smile Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
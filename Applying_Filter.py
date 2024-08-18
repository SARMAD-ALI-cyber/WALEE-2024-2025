import cv2
import numpy as np
import datetime

# Load the Haar cascade file
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#  funny glasses image with transparency (PNG)
glasses = cv2.imread('glasses.png', cv2.IMREAD_UNCHANGED) # so basically this cv2.IMREAD_UNCHANGED argument actually loads the image with the alpha channel

def apply_glasses(frame, x, y, w, h):
    # Resize the glasses image to fit the width of the face
    resized_glasses = cv2.resize(glasses, (w, int(glasses.shape[0] * w / glasses.shape[1])))

    # Calculate position to overlay the glasses (centered on the face)
    y1, y2 = y + int(h / 4), y + int(h / 4) + resized_glasses.shape[0]
    x1, x2 = x, x + resized_glasses.shape[1]

    # Ensure the coordinates are within the frame bounds
    if y1 < 0 or y2 > frame.shape[0] or x1 < 0 or x2 > frame.shape[1]:
        return frame

    # Extract the region of interest (ROI) from the frame where glasses will be placed
    roi = frame[y1:y2, x1:x2]

    # Separate the color and alpha channels
    glasses_rgb = resized_glasses[:, :, :3]
    alpha_channel = resized_glasses[:, :, 3]

    # Create the mask and inverse mask
    mask = cv2.cvtColor(alpha_channel, cv2.COLOR_GRAY2BGR)
    mask = mask / 255.0
    mask_inv = 1.0 - mask

    # Blend the ROI and the glasses images using the masks
    roi = roi * mask_inv + glasses_rgb * mask
    frame[y1:y2, x1:x2] = roi.astype(np.uint8)

    return frame


cap = cv2.VideoCapture(0)
apply_filter_flag = False
recording = False
out = None

while True:
    
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale (required for Haar cascades)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    '''
    scaleFactor-This tells how much the object's size is reduced in each image.
    minNeighbors-This parameter tells how many neighbours each rectangle candidate should consider.
    minSize—This signifies the minimum possible size of an object to be detected. An object smaller than minSize would be ignored.
    '''
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the faces and apply the filter if flag is set
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if apply_filter_flag:
            frame = apply_glasses(frame, x, y, w, h)

    # If recording, write the frame
    if recording and out is not None:
        out.write(frame)

    # Display the resulting frame
    cv2.imshow('Face Detection with Funny Glasses', frame)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF

    # Toggle filter application on 'f' key press
    if key == ord('f'):
        apply_filter_flag = not apply_filter_flag

    # Capture image on 'c' key press
    if key == ord('c'):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        cv2.imwrite(f'captured_{timestamp}.png', frame)
        print(f'Image captured as captured_{timestamp}.png')

    # Start/stop recording on 'r' key press
    if key == ord('r'):
        if not recording:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            out = cv2.VideoWriter(f'recording_{timestamp}.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame.shape[1], frame.shape[0]))
            recording = True
            print(f'Recording started: recording_{timestamp}.avi')
        else:
            recording = False
            out.release()
            out = None
            print('Recording stopped')

    # Break the loop on 'q' key press
    if key == ord('q'):
        break

# Release the webcam and close the window
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()

# Import modules
from collections import deque
import cv2
import time

# Import cascade file
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Import matchlab icon
image = cv2.imread('match_lab_logo1.png', cv2.IMREAD_UNCHANGED)
mask = cv2.cvtColor(image[:, :, -1], cv2.COLOR_GRAY2BGR)
overlay = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) * (mask / 255)

# Turn on camera
videoCapture = cv2.VideoCapture(0)
frameHeight = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
frameWidth = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
minSize = min(frameHeight, frameWidth)

# Preallocate video frame queue
queue = deque()

# Start timer for video recording
start = time.time()
recordingTime = 2
video = 0

# Face tracking
while True:

    # Framerate timer
    frameTimer = time.time()

    # Read in mirror image video frame
    ret, frame = videoCapture.read()
    frame = cv2.flip(frame, 1)

    # Convert to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in gray scale
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3,
                                         minSize=(minSize // 3, minSize // 3),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

    # Add largest face to queue if detected
    if len(faces):

        # Save largest face detected
        face = max(faces, key=lambda i: i[-1] * i[-2])
        queue.appendleft(face)

        # Average up to five video frames
        if len(queue) > 5:
            queue.pop()

    # Reset timer if no face detected
    else:
        start = time.time()

    # Check that at least one video frame is in queue
    if queue:

        # Average bounding box for all frames in queue
        x, y, w, h = sum(queue) // len(queue)

        # Measure brightness of central region of bounding box
        innerBox = (slice(int(y + (0.25 * h)), int(y + (0.75 * h))),
                    slice(int(x+(0.25*w)), int(x+(0.75*w))))
        brightness = int(frame[innerBox[0], innerBox[1], :].mean())

        # Draw bounding box
        length = 50
        lineWidth = 5
        color = (255, 0, 255)
        cv2.line(frame, (x, y), (x+length, y), color, lineWidth)
        cv2.line(frame, (x, y), (x, y+length), color, lineWidth)
        cv2.line(frame, (x+w, y), (x+w-length, y), color, lineWidth)
        cv2.line(frame, (x+w, y), (x+w, y+length), color, lineWidth)
        cv2.line(frame, (x, y+h), (x, y+h-length), color, lineWidth)
        cv2.line(frame, (x, y+h), (x+length, y+h), color, lineWidth)
        cv2.line(frame, (x+w, y+h), (x+w, y+h-length), color, lineWidth)
        cv2.line(frame, (x+w, y+h), (x+w-length, y+h), color, lineWidth)

        # Brightness warning and reset timer
        if brightness < 100:
            cv2.putText(frame, text='Too Dark', org=(50, frameHeight - 50), thickness=4,
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=color)
            start = time.time()
        elif brightness > 180:
            cv2.putText(frame, text='Too Bright', org=(50, frameHeight - 50), thickness=4,
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=color)
            start = time.time()

        # Distance warning and reset timer
        if w < 0.6 * minSize or h < 0.6 * minSize:
            cv2.putText(frame, text='Too Far', org=(frameWidth // 2 + 300, frameHeight - 50),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=color,
                        thickness=4)
            start = time.time()
        elif w > 0.8 * minSize or h > 0.8 * minSize:
            cv2.putText(frame, text='Too Close', org=(frameWidth // 2 + 300, frameHeight - 50),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=color,
                        thickness=4)
            start = time.time()

        # Overlay resizing
        overlayDimensions = tuple(int(0.25 * w * x / image.shape[1]) for x in image.shape[1::-1])
        maskResized = cv2.resize(mask, overlayDimensions)
        overlayResized = cv2.resize(overlay, overlayDimensions)

        # Add overlay
        overlayBox = (slice(y + h - overlayDimensions[1] - 15, y + h - 15),
                      slice(x + 15, x + overlayDimensions[0] + 15))
        frame[overlayBox] = (frame[overlayBox] * (cv2.bitwise_not(maskResized) / 255)).astype('uint8')
        frame[overlayBox] += overlayResized.astype('uint8')

        # Record video
        if time.time() - start > 2:
            video = video if video > 0 else 3 + recordingTime

        if video > 2 + recordingTime:
            cv2.putText(frame, text='3', org=(frameWidth // 2 - 30, 80), thickness=4,
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=3, color=color)
        elif video > 1 + recordingTime:
            cv2.putText(frame, text='2', org=(frameWidth // 2 - 30, 80), thickness=4,
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=3, color=color)
        elif video > 0 + recordingTime:
            cv2.putText(frame, text='1', org=(frameWidth // 2 - 30, 80), thickness=4,
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=3, color=color)
        elif video > 0:
            cv2.putText(frame, text='RECORDING', org=(frameWidth // 2 - 260, 80), thickness=4,
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=3, color=color)

        # Video recording countdown
        video -= time.time() - frameTimer

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit by pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When finished, release the capture
videoCapture.release()
cv2.destroyAllWindows()

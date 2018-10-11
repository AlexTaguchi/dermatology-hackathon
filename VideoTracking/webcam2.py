# !/usr/bin/env python

# Import modules
from collections import deque
import cv2
import dlib
import face_recognition
import numpy as np
import pickle
import time
from skimage.feature import haar_like_feature_coord
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
# from dask import delayed

# 3D model points.
model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corne
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
])

# Import opencv cascade file
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Import dlib face alignment file
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
FULL_POINTS = list(range(0, 68))
FACE_POINTS = list(range(17, 68))
JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Gender and ethnicity detection
ETHNICITIES = 'AWB'
with open('face_model.pkl', 'rb') as file:
    clf, labels = pickle.load(file, encoding='latin1')

# Glasses detection
def extract_feature_image(img, feature_type, feature_coord=None):
    """Extract the haar feature for the current image"""
    ii = integral_image(img)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                             feature_type=feature_type,
                             feature_coord=feature_coord)

with open('glasses_model.pkl', 'rb') as file:
    glasses_haar, idx = pickle.load(file)
feature_types = ['type-2-x', 'type-2-y']
feature_coord, feature_type = \
        haar_like_feature_coord(width=32, height=32,
                                feature_type=feature_types)
selected_feature_coord = feature_coord[idx]
selected_feature_type = feature_type[idx]

# Import matchlab icon
image = cv2.imread('match_lab_logo.png', cv2.IMREAD_UNCHANGED)
mask = cv2.cvtColor(image[:, :, -1], cv2.COLOR_GRAY2BGR)
overlay = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) * (mask / 255)

# Turn on camera
videoCapture = cv2.VideoCapture(0)
frameHeight = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
frameWidth = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
minSize = min(frameHeight, frameWidth)

# Preallocate video frame and ethnicity/gender queues
videoQueue = deque()
egQueue = deque()

# Start timer for video recording
timer = time.time()

# Video recording time in seconds
recording = 10

# Wait time in seconds between video recordings
wait = 3

# Preallocate timing, redness, glasses, and gender/ethnicity counter variables
video = 0
redness = []
# glasses = []
egCounter = 2

# Face tracking
while True:

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
        videoQueue.appendleft(face)

        # Average up to five video frames
        if len(videoQueue) > 5:
            videoQueue.pop()

    # Reset timer if no face detected
    else:
        timer = time.time()

    # Check that at least one video frame is in queue
    if videoQueue:
        # Average bounding box for all frames in queue
        x, y, w, h = sum(videoQueue) // len(videoQueue)

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
            timer = time.time()
        elif brightness > 180:
            cv2.putText(frame, text='Too Bright', org=(50, frameHeight - 50), thickness=4,
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=color)
            timer = time.time()

        # Distance warning and reset timer
        if w < 0.6 * minSize or h < 0.6 * minSize:
            cv2.putText(frame, text='Too Far', org=(frameWidth // 2 + 300, frameHeight - 50),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=color,
                        thickness=4)
            timer = time.time()
        elif w > 0.8 * minSize or h > 0.8 * minSize:
            cv2.putText(frame, text='Too Close', org=(frameWidth // 2 + 300, frameHeight - 50),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=color,
                        thickness=4)
            timer = time.time()

        # MatchLab icon overlay resizing
        overlayDimensions = tuple(int(0.25 * w * x / image.shape[1]) for x in image.shape[1::-1])
        maskResized = cv2.resize(mask, overlayDimensions)
        overlayResized = cv2.resize(overlay, overlayDimensions)

        # Add MatchLab icon overlay
        overlayBox = (slice(y + h - overlayDimensions[1] - 15, y + h - 15),
                      slice(x + 15, x + overlayDimensions[0] + 15))
        frame[overlayBox] = (frame[overlayBox] * (cv2.bitwise_not(maskResized) / 255)).astype('uint8')
        frame[overlayBox] += overlayResized.astype('uint8')

        # List gender and ethnicity
        if egCounter > 1:
            face_encodings = face_recognition.face_encodings(frame, known_face_locations=[(y, x+w, y+h, x)])
            prediction = clf.predict_proba(face_encodings[0].reshape(1, -1))[0][:4]
            egQueue.appendleft(prediction)
            egCounter -= 1
        else:
            egCounter += 1

        # Display gender and ethnicity
        if len(egQueue) >= 20:
            egAverage = [sum(x) / 20 for x in zip(*egQueue)]
            egAverage[1:] = [x / sum(egAverage[1:]) for x in egAverage[1:]]
            gender = 'Male' if egAverage[0] >= 0.5 else 'Female'

            # Display gender
            cv2.putText(frame, text=gender, org=(x + w - 100, y + h + 40),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.2, color=color, thickness=4)

            # Display ethnicity
            cv2.putText(frame, text=ETHNICITIES, org=(x + w - 100, y + h - 20),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.2, color=color, thickness=4)
            cv2.rectangle(frame, (x + w - 99, y + h - 60 - int(100 * egAverage[1])),
                          (x + w - 75, y + h - 60), (0, 0, 255), cv2.FILLED)
            cv2.rectangle(frame, (x + w - 72, y + h - 60 - int(100 * egAverage[2])),
                          (x + w - 48, y + h - 60), (0, 255, 0), cv2.FILLED)
            cv2.rectangle(frame, (x + w - 45, y + h - 60 - int(100 * egAverage[3])),
                          (x + w - 21, y + h - 60), (255, 0, 0), cv2.FILLED)

        # Glasses recording
        # face_region = cv2.resize(gray[y:y + h, x:x + w], (32, 32), interpolation=cv2.INTER_LINEAR)
        # cv2.imwrite('Alex/negative/%d.jpg' % (10*time.time()), face_region)
        # cv2.imwrite('Alex/positive/%d.jpg' % (10*time.time()), face_region)

        # Pop end of egQueue if too long
        if len(egQueue) > 20:
            egQueue.pop()

        # Record video
        if time.time() - timer > 2 and not video:
            video = time.time() + 3 + recording + wait

        if video - time.time() > 2 + recording + wait:
            # Check for glasses
            # face_region = cv2.resize(gray[y:y + h, x:x + w], (32, 32), interpolation=cv2.INTER_LINEAR)
            # X = extract_feature_image(face_region, selected_feature_type, selected_feature_coord)
            # glasses_prob = glasses_haar.predict_proba(np.array(X).reshape(1, -1))
            # glasses.append(np.argmax(glasses_prob[0]))

            cv2.putText(frame, text='3', org=(frameWidth // 2 - 30, 80), thickness=4,
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=3, color=color)
        elif video - time.time() > 1 + recording + wait:
            # Check for glasses
            # face_region = cv2.resize(gray[y:y + h, x:x + w], (32, 32), interpolation=cv2.INTER_LINEAR)
            # X = extract_feature_image(face_region, selected_feature_type, selected_feature_coord)
            # glasses_prob = glasses_haar.predict_proba(np.array(X).reshape(1, -1))
            # glasses.append(np.argmax(glasses_prob[0]))

            cv2.putText(frame, text='2', org=(frameWidth // 2 - 30, 80), thickness=4,
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=3, color=color)
        elif video - time.time() > 0 + recording + wait:
            # Check for glasses
            # face_region = cv2.resize(gray[y:y + h, x:x + w], (32, 32), interpolation=cv2.INTER_LINEAR)
            # X = extract_feature_image(face_region, selected_feature_type, selected_feature_coord)
            # glasses_prob = glasses_haar.predict_proba(np.array(X).reshape(1, -1))
            # glasses.append(np.argmax(glasses_prob[0]))

            cv2.putText(frame, text='1', org=(frameWidth // 2 - 30, 80), thickness=4,
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=3, color=color)
        elif video - time.time() > wait:
            # Check for glasses
            # if sum(glasses) / len(glasses) < 0.5:
            #
            #     # Remove glasses
            #     cv2.putText(frame, text='Remove Glasses', org=(250, 80), thickness=4,
            #                 fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=3, color=color)
            #
            # else:

                # Detect dlib face rectangles
                factor = 4
                gray = cv2.resize(gray, None, fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_LINEAR)
                rectangles = detector(gray, 0)

                # Track face features if bounding box detected
                if rectangles:
                    # Face shape prediction
                    shape = predictor(gray, rectangles[0])
                    coordinates = np.zeros((shape.num_parts, 2), dtype='int')
                    for x in range(0, shape.num_parts):
                        coordinates[x] = (shape.part(x).x, shape.part(x).y)
                    shape = factor * coordinates

                    # Forehead top and side anchors
                    forehead_rt = 2 * (shape[19] - shape[36]) + shape[19]
                    forehead_lt = 2 * (shape[24] - shape[45]) + shape[24]
                    forehead_rs = 2 * (shape[19] - shape[36]) + shape[0]
                    forehead_ls = 2 * (shape[24] - shape[45]) + shape[16]

                    # Forehead anchor midpoints
                    midpoint_r = [0.25 * (forehead_rt[0] - forehead_rs[0]) + forehead_rs[0],
                                  0.75 * (forehead_rt[1] - forehead_rs[1]) + forehead_rs[1]]
                    midpoint_l = [0.25 * (forehead_lt[0] - forehead_ls[0]) + forehead_ls[0],
                                  0.75 * (forehead_lt[1] - forehead_ls[1]) + forehead_ls[1]]

                    # Add forehead anchor points
                    shape = np.vstack((shape, forehead_rt, forehead_lt,
                                       forehead_rs, forehead_ls,
                                       midpoint_r, midpoint_l)).astype(np.int)

                    # Preallocate mask array
                    feature_mask = np.zeros((frame.shape[0], frame.shape[1]))

                    # Facial areas
                    face_forehead = cv2.convexHull(shape)
                    eye_right = cv2.convexHull(shape[RIGHT_EYE_POINTS])
                    eye_left = cv2.convexHull(shape[LEFT_EYE_POINTS])
                    mouth = cv2.convexHull(shape[MOUTH_OUTLINE_POINTS])

                    # Generate face mask
                    cv2.fillConvexPoly(feature_mask, face_forehead, 1)
                    cv2.fillConvexPoly(feature_mask, eye_right, 0)
                    cv2.fillConvexPoly(feature_mask, eye_left, 0)
                    cv2.fillConvexPoly(feature_mask, mouth, 0)
                    feature_mask = feature_mask.astype(np.bool)

                    # Frame redness: red - max(green, blue)
                    frame_red = frame[:, :, 2].astype(np.int)
                    frame_red -= np.max(frame[:, :, :2], axis=-1).astype(np.int)

                    # Face redness: red - max(green, blue)
                    face_red = frame[feature_mask, 2].astype(np.int)
                    face_red -= np.max(frame[feature_mask, :2], axis=-1).astype(np.int)

                    # Median normalize the red intensities
                    frame_red = (128 / np.median(face_red)) * frame_red
                    frame_red[frame_red < 0] = 0
                    frame_red[frame_red > 255] = 255

                    # Remove areas less than half the median redness from the mask
                    feature_mask[frame_red < 64] = False

                    # Overlay mask as a heat map with the jet color scheme
                    frame_jet = frame_red.astype(np.uint8)
                    frame_jet = cv2.applyColorMap(frame_jet, cv2.COLORMAP_JET)
                    frame[feature_mask] = 0.8 * frame[feature_mask] + (0.2 * frame_jet[feature_mask]).astype(np.uint8)

                    # Calculate average redness score for pixels above the median redness
                    feature_mask[frame_red < 128] = False
                    redness.append(np.mean(frame[feature_mask, 2]))
                    statistics = np.mean(redness).item(), np.std(redness).item()
                    cv2.putText(frame, text='Recording', org=(400, 80), thickness=4,
                                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=3, color=color)
                    # cv2.putText(frame, text='Redness: %.1f +/- %.1f' % statistics, org=(30, 80), thickness=4,
                    #             fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=3, color=color)

                    ###

                    # 2D image points. If you change the image, you need to change vector
                    image_points = np.array([
                        shape[30],  # Nose tip
                        shape[8],  # Chin
                        shape[45],  # Left eye left corner
                        shape[36],  # Right eye right corner
                        shape[54],  # Left Mouth corner
                        shape[48]  # Right mouth corner
                    ], dtype="double")

                    # Camera internals
                    size = frame.shape
                    focal_length = size[0]
                    center = (size[1] / 2, size[0] / 2)
                    camera_matrix = np.array(
                        [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype="double"
                    )

                    # print("Camera Matrix :\n {0}".format(camera_matrix))

                    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
                    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points,
                                                                                  camera_matrix,
                                                                                  dist_coeffs,
                                                                                  flags=cv2.SOLVEPNP_ITERATIVE)

                    # print("Rotation Vector:\n {0}".format(rotation_vector))
                    # print("Translation Vector:\n {0}".format(translation_vector))

                    # Project a 3D point (0, 0, 1000.0) onto the image plane.
                    # We use this to draw a line sticking out of the nose

                    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector,
                                                                     translation_vector,
                                                                     camera_matrix, dist_coeffs)

                    for p in image_points:
                        cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

                    p1 = (int(image_points[0][0]), int(image_points[0][1]))
                    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

                    cv2.line(frame, p1, p2, (255, 0, 0), 2)
                    ###

        elif video - time.time() > 0:
            pass

        else:
            # Reset video and redness variables
            video = 0
            redness = []
            # glasses = []

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit by pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When finished, release the capture
videoCapture.release()
cv2.destroyAllWindows()

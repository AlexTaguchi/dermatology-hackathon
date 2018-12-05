# Hacking Dermatology 2018
High-throughput dermatological data collection, labeling, and redness detection

Labeling Tool
==========
GUI interface for rapid labeling of dermatologically relevant images.

Usage:
- Download the full contents of the LabelingTool folder and run "python DemoLabeler.py"
- The image file name and the remaining number of unlabeled images are indicated at the top
- Accept or reject the image
- Use the buttons below to provide reasons for the decision
- Click Submit to save your rating to a text file.
- Click Skip to skip irrelevant images
- (Optional) Include additional information about the image in the Notes form field section

(Module dependencies: tkinter, google_images_download, PIL)

Redness Detection
==========
Identifies afflicted red regions of a person's face and provides an overall redness score

Usage:
- Download the full contents of the RednessDetection folder
- Download pretrained Keras weights Keras_FCN8s_face_seg_YuvalNirkin.h5 into this folder (https://drive.google.com/uc?id=1alyR6uv4CHt1WhykiQIiK5MZir7HSOUU&export=download)
- Run the python script with an image of a person's face as input (for example, "python Red.py 4.jpg")
- Outputs image of face with afflicted region mask, and prints redness score

(Module dependencies: opencv-python, keras, tensorflow)

Video Tracking
==========
Collect face images/video adhering to strict lighting and resolution metrics

Usage:
- Download the full contents of the VideoTracking folder
- Run "python webcam.py"
- The webcam will record photo/video of the face once lighting and face resolution criteria are met

(Module dependencies: webcam.py)

# Dermatology
High-throughput labeling and redness detection of dermatological images

Labeling Tool
==========
GUI interface for rapid labeling of dermatologically relevant images.

Usage:
- Download the full contents of the LabelingTool folder and run "python DermLabel.py" in this directory
- (Optional) Identify yourself by typing your name in the upper right form field
- The second line of indicates the image file name and the remaining number of unlabeled images
- Rate the image on a scale of 0-9 on Redness, Texture, and Evenness
- Click Submit to save your rating to a text file.
- Click Skip to skip irrelevant images
- (Optional) Include additional information about the image in the Tag form field section
- (Optional) Download additional images for labeling by providing an image URL in the Download form field (20 visually similar images will be downloaded

(Module dependencies: tkinter, google_images_download, PIL)

Redness Detection
==========
Identifies afflicted red regions of a person's face and provides an overall redness score

Usage:
- Download the full contents of the RednessDetection folder
- Download pretrained Keras weights Keras_FCN8s_face_seg_YuvalNirkin.h5 into this folder (https://drive.google.com/uc?id=1alyR6uv4CHt1WhykiQIiK5MZir7HSOUU&export=download)
- Run the python script with an image of a person's face as input (for example, "python Red.py 4.jpg")
- Outputs image of face with afflicted region mask, and prints redness score

(Module dependencies: cv2, keras, tensorflow)

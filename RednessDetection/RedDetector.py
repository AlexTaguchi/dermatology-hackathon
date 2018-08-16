# Redness Detection Algorithm Utilizing AI Face Detection

# Face detection algorithm: https://github.com/shaoanlu/face-segmentation-keras
# Keras weights: https://drive.google.com/uc?id=1alyR6uv4CHt1WhykiQIiK5MZir7HSOUU&export=download

# Import modules
import cv2
from keras.layers import *
import sys
from FCN8s_keras import FCN


# Main function
def main(photo):

    # Instantiate face detection neural network and load pretrained weights
    model = FCN()
    model.load_weights("Keras_FCN8s_face_seg_YuvalNirkin.h5")

    # ~~~Process Image~~~ #
    def preprocess(image):

        # Resize image to 500x500 pixels
        image = cv2.resize(image, (500, 500))

        # Arbitrary color scaling that makes the algorithm work better on select data
        image = image.astype(float) - np.array((122.67891434, 116.66876762, 104.00698793))

        # Convert image to convention required by face detection algorithm
        image = image[np.newaxis, :, :, ::-1]

        return image


    # Pass image through face detection algorithm
    im = cv2.cvtColor(cv2.imread(photo), cv2.COLOR_BGR2RGB)
    image_in = preprocess(im)
    out = model.predict(image_in)

    # Generate mask for face detection
    out_resized = cv2.resize(np.squeeze(out), (im.shape[1], im.shape[0]))
    out_resized_clipped = np.clip(out_resized.argmax(axis=2), 0, 1).astype(np.float64)
    face_mask = cv2.GaussianBlur(out_resized_clipped, (7, 7), 6)

    # Redness criterion of each pixel (absolute)
    red_absolute = 200 > im[:, :, 0]

    # Redness criterion of each pixel (relative)
    red_relative = im[:, :, 0] > (0.9 * (im[:, :, 1] + im[:, :, 2]))

    # Pixels must both have a high red value and relatively lower blue and green values
    red_spots = np.logical_and(red_absolute, red_relative)

    # Generate mask of red regions from face mask
    red_mask = np.copy(face_mask)
    red_mask[np.invert(red_spots)] = 0
    red_mask[red_mask != 1] = 0

    # Calculate redness score
    face_pixels = sum(face_mask.reshape(-1) > 0)
    red_pixels = sum(red_mask.reshape(-1) > 0)
    red_score = 1000 * red_pixels / face_pixels

    # Generate labeled image
    overlay = np.copy(im)
    overlay[red_mask == 1, 0] = 255
    overlay[red_mask == 1, 1] = 0
    overlay[red_mask == 1, 2] = 255
    im_mask = cv2.addWeighted(im, 0.5, overlay, 0.5, 0)

    # Output image and redness score
    cv2.imwrite(photo[:-4] + '_labeled' + photo[-4:], im_mask[:, :, ::-1])
    print('Redness Score: %d' % red_score)
    return red_score

if __name__ == "__main__":
    main(sys.argv[1])

import cv2
import mlrose
import numpy as np
from PIL import Image
from image_generation.HogFace import HOGFace
from facial_detection.face_detector import detect_faces

OUTPUT_DIR = "../output/"
OUTPUT_FILE_NAME = "image.png"
OUTPUT = OUTPUT_DIR + OUTPUT_FILE_NAME

# arbitrarily selected as (4*hog_width, 4*hog_height)
width, height = (1532, 1528)
# each column represents a HogFace
# arbitrarily trying with 6 faces because mlrose expects known problem length
initial_state = np.array([[0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [256, 0, 0, 0, 0, 0],
                          [256, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 0]])


# TODO: remove
def testing_img():
    img = get_background_img()

    for col in np.nditer(initial_state, flags=['external_loop'], order='F'):
        pos_x = col[0]
        pos_y = col[1]
        rot = col[2]
        size_x = col[3]
        size_y = col[4]
        display = col[5]

        hog_img = HOGFace((pos_x, pos_y), rot, (size_x, size_y), display)

        if hog_img.display and size_x > 0 and size_y > 0:
            img.paste(hog_img.get_image(), hog_img.position)

    img.save(OUTPUT, "PNG")
    print(len(detect_faces(cv2.imread(OUTPUT))))


def get_background_img():
    background = Image.new('RGB', (width, height), (0, 0, 0))
    return background


def detected_max(state):
    img = get_background_img()
    # TODO: reformat for state
    for col in np.nditer(state, flags=['external_loop'], order='F'):
        pos_x = col[0]
        pos_y = col[1]
        rot = col[2]
        size_x = col[3]
        size_y = col[4]
        display = col[5]

        hog_img = HOGFace((pos_x, pos_y), rot, (size_x, size_y), display)

        if hog_img.display and size_x > 0 and size_y > 0:
            img.paste(hog_img.get_image(), hog_img.position)

    img.save(OUTPUT, "PNG")
    return len(detect_faces(cv2.imread(OUTPUT)))


def main():
    # TODO: remove
    testing_img()

    # want to maximize this
    fitness = mlrose.CustomFitness(detected_max)


if __name__ == "__main__":
    # TODO: take output name in input args
    main()

import cv2
from PIL import Image
from image_generation.HogFace import HOGFace
from facial_detection.face_detector import detect_faces

OUTPUT_DIR = "../output/"
OUTPUT_FILE_NAME = "image.png"
OUTPUT = OUTPUT_DIR + OUTPUT_FILE_NAME

# arbitrarily selected as (4*hog_width, 4*hog_height)
width, height = (1532, 1528)


def get_background_img():
    background = Image.new('RGB', (width, height), (0, 0, 0))
    return background


def main():
    img = get_background_img()

    hog_img = HOGFace(0, (0, 0), (0, 0, 0))
    img.paste(hog_img.img, hog_img.position)

    img.save(OUTPUT, "PNG")

    # need to read the image after saving so that its the right format for opencv
    print(len(detect_faces(cv2.imread(OUTPUT))))


if __name__ == "__main__":
    # TODO: take output name in input args
    main()

import cv2
from PIL import Image
from facial_detection.face_detector import detect_faces

OUTPUT_DIR = "../output/"
OUTPUT_FILE_NAME = "image.png"
OUTPUT = OUTPUT_DIR + OUTPUT_FILE_NAME
HOG_FILE = "../data/hog_no_bg.png"

hog_img = Image.open(HOG_FILE)
width, height = hog_img.size


def get_base_img():
    img = Image.new('RGB', (width * 4, height * 4), (0, 0, 0))
    return img


def main():
    img = get_base_img()

    # fill image
    for i in range (0, 4):
        for j in range (0, 4):
            img.paste(hog_img, (width*i, height*j))

    img.save(OUTPUT, "PNG")

    # need to read the image after saving so that its the right format for opencv
    print(len(detect_faces(cv2.imread(OUTPUT))))


if __name__ == "__main__":
    # TODO: take output name in input args
    main()

from PIL import Image

OUTPUT_DIR = "../output/"
OUTPUT_FILE_NAME = "image.png"
OUTPUT = OUTPUT_DIR + OUTPUT_FILE_NAME
HOG_FILE = "../data/hog_no_bg.png"


def main():
    # https://www.geeksforgeeks.org/working-images-python/
    hog_img = Image.open(HOG_FILE)
    width, height = hog_img.size

    img = Image.new('RGB', (width*4, height*4), (0, 0, 0))
    img.paste(hog_img)
    img.save(OUTPUT, "PNG")


if __name__ == "__main__":
    # TODO: take output name in input args
    main()

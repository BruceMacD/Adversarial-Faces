from PIL import Image


class HOGFace:

    HOG_FILE = "../data/hog_no_bg.png"

    img = Image.open(HOG_FILE)
    width, height = img.size

    def __init__(self, rotation, position, color):
        self.rotation = rotation
        self.position = position
        self.color = color

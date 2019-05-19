from PIL import Image


class HOGFace:

    HOG_FILE = "../data/hog_no_bg.png"

    img = Image.open(HOG_FILE)
    width, height = img.size

    def __init__(self, position, rotation, size, display):
        self.position = position
        self.rotation = rotation
        self.size = size
        # controls if image should be displayed, int because of mlrose assumptions
        self.display = display

    # apply all features to image and return
    def get_image(self):
        # TODO: rotation
        im = self.img
        im.thumbnail(self.size, Image.ANTIALIAS)
        return im

    def display_image(self):
        if self.display == 0:
            return False
        return True

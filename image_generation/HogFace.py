from PIL import Image


class HOGFace:

    HOG_FILE = "../data/hog_no_bg.png"

    img = Image.open(HOG_FILE)
    width, height = img.size

    def __init__(self, position, rotation, scale, scale_factor):
        self.position = position
        self.rotation = rotation%360
        # scale will come in range
        self.scale = scale

    # apply all features to image and return
    def get_image(self):
        size = (self.width*self.scale, self.height*self.scale)
        if 0 in size:
            # do not display this image
            return None
        im = self.img
        # change the size of the image
        im.thumbnail(size, Image.ANTIALIAS)
        im = im.rotate(self.rotation)
        return im

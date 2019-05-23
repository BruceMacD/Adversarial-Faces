import cv2
import mlrose
import numpy as np
from PIL import Image
from image_generation.HogFace import HOGFace
from facial_detection.face_detector import detect_faces

OUTPUT_DIR = "../output/"
OUTPUT_FILE_NAME = "image.png"
OUTPUT = OUTPUT_DIR + OUTPUT_FILE_NAME

# arbitrarily selected as (2*hog_width, 2*hog_height), smaller than the amount of face input in standard dimensions
width, height = (766, 766)
scale_factor = 766
# constrained by mlrose to 1D array, so need to workaround with index pos significance
# arbitrarily trying with 6 faces because mlrose expects known problem length
# indexes:
# 0 % 4 = pos_x
# 1 % 4 = pos_y
# 2 % 4 = rotation
# 3 % 4 = scale
initial_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# we know that one good state is to have 4 HOGs taking the whole image, lets start there


def get_background_img():
    background = Image.new('RGB', (width, height), (0, 0, 0))
    return background


def get_img_from_state(state):
    img = get_background_img()

    # use state as coordinates of faces in image and params
    i = 0
    while i < len(state):
        pos_x = state[i]
        pos_y = state[i + 1]
        rotation = state[i + 2]
        scale = state[i + 3]
        hog_img = HOGFace((pos_x, pos_y), rotation, scale, scale_factor)
        result = hog_img.get_image()
        if result is not None:
            img.paste(result, hog_img.position)
        i = i + 4

    img.save(OUTPUT, "PNG")


def detected_max(state):
    get_img_from_state(state)
    # currently fitness is only based on number of faces detected.
    # Biasing fitness for larger input faces could be ideal to ensure detection from a distance.
    return len(detect_faces(cv2.imread(OUTPUT)))


def main():
    # want to maximize this
    fitness = mlrose.CustomFitness(detected_max)
    problem = mlrose.DiscreteOpt(length=24, fitness_fn=fitness,
                                 maximize=True, max_val=scale_factor)
    schedule = mlrose.ExpDecay()
    best_state, max_faces = mlrose.simulated_annealing(problem, schedule=schedule, max_attempts=10, max_iters=1000,
                                                       init_state=initial_state, random_state=1)

    print('Optimal state found: ', best_state)
    print('Max fitness found: ', max_faces)
    # save the optimal found
    get_img_from_state(best_state)
    print("Number of faces in output: ", len(detect_faces(cv2.imread(OUTPUT))))


if __name__ == "__main__":
    main()

import os
from collections import Counter

# import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# CONF = config.get_conf_dict()
homedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# base_dir = CONF['general']['base_directory']
base_dir = "."

model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
    device
)


def ground_dino():
    return model, processor


from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
text_labels = ["green region. foliage."]


def sam2():
    return predictor, text_labels


def minimum_betw_max(dico_, visua=False):
    Ax = list(dico_.keys())
    Ay = list(dico_.values())

    # Approximation par une régression polynomiale
    x = Ax[1:]
    y = Ay[1:]
    degree = 14  # Choisissez le degré selon la complexité de la courbe
    coefficients = np.polyfit(x, y, degree)
    polynomial = np.poly1d(coefficients)

    # Points lissés pour tracer la courbe
    x_fit = np.linspace(min(x), max(x), 500)
    y_fit = polynomial(x_fit)

    # Détection des maxima
    peaks, _ = find_peaks(y_fit)

    peak_values = y_fit[peaks]
    sorted_indices = np.argsort(peak_values)[
        ::-1
    ]  # Trier en ordre décroissant
    top_two_peaks = peaks[
        sorted_indices[:2]
    ]  # Les indices des deux plus grands pics

    # Trouver le minimum entre les deux maxima
    x_min_range = x_fit[top_two_peaks[0] : top_two_peaks[1] + 1]
    y_min_range = y_fit[top_two_peaks[0] : top_two_peaks[1] + 1]
    minx = min([top_two_peaks[0], top_two_peaks[1]])
    maxx = max([top_two_peaks[0], top_two_peaks[1]])
    x_min_range = x_fit[minx : maxx + 1]
    y_min_range = y_fit[minx : maxx + 1]
    min_index = np.argmin(y_min_range)  # Index du minimum dans cette plage
    x_min = x_min_range[min_index]
    y_min = y_min_range[min_index]

    # if visua:
    #     # Tracé
    #     plt.scatter(x, y, color="blue")
    #     plt.plot(x_fit, y_fit, color="red", label="Polynomial regression")
    #     plt.scatter(
    #         x_fit[top_two_peaks],
    #         y_fit[top_two_peaks],
    #         color="green",
    #         label="Local maximum",
    #     )
    #     plt.scatter(x_min, y_min, color="orange", s=100, label="Local minimum")
    #     plt.legend()
    #     plt.xlabel("Depth pixel")
    #     plt.ylabel("Count")
    #     # plt.title('Approximation et détection des points maximum')
    #     plt.show()
    return x_min, y_min


def frontground_part(depths):
    depth_one = depths[:, :]
    n, m = depth_one.shape
    A = []
    for i in tqdm(range(n)):
        for j in range(m):
            A.append([i, j, depth_one[i, j]])
    X = np.array(A)

    dico_ = Counter(X[:, 2])
    min_coord = minimum_betw_max(dico_, visua=False)

    th_ = min_coord[0]
    msks_depth = depth_one > th_
    return msks_depth

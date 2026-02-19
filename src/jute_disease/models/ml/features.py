from abc import ABC, abstractmethod

import cv2
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from jute_disease.utils.constants import IMAGE_SIZE
from jute_disease.utils.logger import get_logger

logger = get_logger(__name__)


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    @abstractmethod
    def extract(self, img: Image.Image | np.ndarray) -> np.ndarray:
        """Extract features from an image."""
        pass


class HandcraftedFeatureExtractor(BaseFeatureExtractor):
    """Extracts Color (HSV), Texture (LBP, GLCM), and Shape (HOG) features."""

    def __init__(self) -> None:
        self.lbp_radius = 3
        self.lbp_n_points = 8 * self.lbp_radius
        self.lbp_method = "uniform"

        self.glcm_distances = [1, 3]
        self.glcm_angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        self.glcm_props = [
            "contrast",
            "dissimilarity",
            "homogeneity",
            "energy",
            "correlation",
            "ASM",
        ]

        self.hog_orientations = 9
        self.hog_pixels_per_cell = (16, 16)
        self.hog_cells_per_block = (2, 2)

    def extract(self, img: Image.Image | np.ndarray) -> np.ndarray:
        """Extract combined color and texture feature vector."""
        if isinstance(img, Image.Image):
            img = np.array(img)

        img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        h_mean, h_std = np.mean(hsv[:, :, 0]), np.std(hsv[:, :, 0])
        s_mean, s_std = np.mean(hsv[:, :, 1]), np.std(hsv[:, :, 1])
        v_mean, v_std = np.mean(hsv[:, :, 2]), np.std(hsv[:, :, 2])
        color_features = np.array([h_mean, h_std, s_mean, s_std, v_mean, v_std])

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(
            gray, self.lbp_n_points, self.lbp_radius, self.lbp_method
        )
        n_bins = int(lbp.max() + 1)
        lbp_hist, _ = np.histogram(
            lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True
        )

        glcm = graycomatrix(
            gray,
            distances=self.glcm_distances,
            angles=self.glcm_angles,
            levels=256,
            symmetric=True,
            normed=True,
        )

        glcm_features = []
        for prop in self.glcm_props:
            val = graycoprops(glcm, prop).flatten()
            glcm_features.extend(val)
        glcm_features = np.array(glcm_features)

        hog_features = hog(
            gray,
            orientations=self.hog_orientations,
            pixels_per_cell=self.hog_pixels_per_cell,
            cells_per_block=self.hog_cells_per_block,
            block_norm="L2-Hys",
            transform_sqrt=True,
            feature_vector=True,
        )

        combined = np.hstack([color_features, lbp_hist, glcm_features, hog_features])

        return combined.astype(np.float32)


class RawPixelFeatureExtractor(BaseFeatureExtractor):
    """Extracts raw pixel values as features."""

    def __init__(self, img_size: int = IMAGE_SIZE) -> None:
        self.img_size = img_size

    def extract(self, img: Image.Image | np.ndarray) -> np.ndarray:
        """Resize image and flatten pixel values."""
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        img_resized = img.resize((self.img_size, self.img_size))
        img_array = np.array(img_resized)
        return img_array.flatten().astype(np.float32)


def extract_features(
    dataset: ImageFolder,
    extractor: BaseFeatureExtractor,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features using the provided extractor."""
    features_list = []
    labels_list = []

    logger.info(f"Extracting features using {type(extractor).__name__}...")
    for i in tqdm(range(len(dataset)), desc="Feature Extraction"):
        img, label = dataset[i]
        feats = extractor.extract(img)
        features_list.append(feats)
        labels_list.append(label)

    return np.array(features_list), np.array(labels_list)

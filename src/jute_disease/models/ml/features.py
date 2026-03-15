# ruff: noqa: N806
from abc import ABC, abstractmethod

import cv2
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from skimage.filters import gabor_kernel
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from jute_disease.utils.constants import IMAGE_SIZE, ML_FEATURES_DIR
from jute_disease.utils.logger import get_logger

logger = get_logger(__name__)


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    @abstractmethod
    def extract(self, img: Image.Image | np.ndarray) -> np.ndarray:
        """Extract features from an image."""
        pass


class CraftedFeatureExtractor(BaseFeatureExtractor):
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

        self.gabor_kernels = []
        # Orientations: 0, 45, 90, 135 degrees
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            kernel = cv2.getGaborKernel((21, 21), 5.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            self.gabor_kernels.append(kernel)
        
    def extract(self, img: Image.Image | np.ndarray) -> np.ndarray:
        """Extract combined color and texture feature vector."""
        if isinstance(img, Image.Image):
            img = np.array(img)

        img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        # Color Moments
        h_mean, h_std = np.mean(hsv[:, :, 0]), np.std(hsv[:, :, 0])
        s_mean, s_std = np.mean(hsv[:, :, 1]), np.std(hsv[:, :, 1])
        v_mean, v_std = np.mean(hsv[:, :, 2]), np.std(hsv[:, :, 2])
        color_moments = np.array([h_mean, h_std, s_mean, s_std, v_mean, v_std])

        # Color Histogram (8 bins per channel)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        color_hist = hist.flatten()

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(
            gray, self.lbp_n_points, self.lbp_radius, self.lbp_method
        )
        # Number of bins for 'uniform' LBP is points + 2
        n_bins = self.lbp_n_points + 2
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

        # # Gabor Features
        # gabor_features = []
        # for kernel in self.gabor_kernels:
        #     fimg = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        #     gabor_features.append(np.mean(fimg))
        #     gabor_features.append(np.std(fimg))
        # gabor_features = np.array(gabor_features)
        
        combined = np.hstack(
            [color_moments, color_hist, lbp_hist, glcm_features, hog_features]
        )

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
    cache_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features using the provided extractor, with optional caching."""
    if cache_name is not None:
        ML_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        extractor_name = type(extractor).__name__.lower()
        full_cache_name = f"{extractor_name}_{cache_name}"
        x_path = ML_FEATURES_DIR / f"{full_cache_name}_X.npy"
        y_path = ML_FEATURES_DIR / f"{full_cache_name}_y.npy"

        if x_path.exists() and y_path.exists():
            logger.info(f"Loading cached features: {full_cache_name}")
            return np.load(x_path), np.load(y_path)

    features_list = []
    labels_list = []

    logger.info(f"Extracting features using {type(extractor).__name__}...")
    for i in tqdm(range(len(dataset)), desc="Feature Extraction"):
        img, label = dataset[i]
        feats = extractor.extract(img)
        features_list.append(feats)
        labels_list.append(label)

    X = np.array(features_list)
    y = np.array(labels_list)

    if cache_name is not None:
        np.save(x_path, X)
        np.save(y_path, y)
        logger.info(f"Saved cached features: {full_cache_name}")

    return X, y


FEATURE_EXTRACTORS: dict[str, type[BaseFeatureExtractor]] = {
    "crafted": CraftedFeatureExtractor,
    "raw": RawPixelFeatureExtractor,
}

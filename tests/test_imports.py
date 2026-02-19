"""Validate the public __init__ API surface for each package."""


def test_root_package_exposes_subpackages():
    from jute_disease import data, models, utils

    assert data is not None
    assert models is not None
    assert utils is not None


def test_data_api():
    from jute_disease.data import (
        DataModule,
        PretrainDataModule,
        dl_train_transforms,
        dl_val_transforms,
        ml_train_transforms,
        ml_val_transforms,
    )

    assert DataModule is not None
    assert PretrainDataModule is not None
    assert dl_train_transforms is not None
    assert dl_val_transforms is not None
    assert ml_train_transforms is not None
    assert ml_val_transforms is not None


def test_models_dl_api():
    from jute_disease.models.dl import Classifier, MobileViT, TimmBackbone

    assert Classifier is not None
    assert MobileViT is not None
    assert TimmBackbone is not None


def test_models_ml_api():
    from jute_disease.models.ml import (
        BaseFeatureExtractor,
        HandcraftedFeatureExtractor,
        KNearestNeighbors,
        LogisticRegression,
        MultinomialNaiveBayes,
        RandomForest,
        RawPixelFeatureExtractor,
        SklearnClassifier,
        SupportVectorMachine,
        extract_features,
    )

    assert SklearnClassifier is not None
    assert KNearestNeighbors is not None
    assert LogisticRegression is not None
    assert MultinomialNaiveBayes is not None
    assert RandomForest is not None
    assert SupportVectorMachine is not None
    assert BaseFeatureExtractor is not None
    assert HandcraftedFeatureExtractor is not None
    assert RawPixelFeatureExtractor is not None
    assert extract_features is not None


def test_utils_api():
    from jute_disease.utils import (
        BATCH_SIZE,
        DATA_DIR,
        DEFAULT_SEED,
        IMAGE_SIZE,
        ML_MODELS_DIR,
        ML_SPLIT_DIR,
        ROOT_DIR,
        SPLITS,
        get_logger,
        seed_everything,
        setup_wandb,
        split_data,
    )

    assert ROOT_DIR is not None
    assert DATA_DIR is not None
    assert ML_MODELS_DIR is not None
    assert ML_SPLIT_DIR is not None
    assert isinstance(IMAGE_SIZE, int)
    assert isinstance(BATCH_SIZE, int)
    assert isinstance(DEFAULT_SEED, int)
    assert isinstance(SPLITS, dict)
    assert get_logger is not None
    assert seed_everything is not None
    assert setup_wandb is not None
    assert split_data is not None

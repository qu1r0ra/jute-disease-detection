from jute_disease.utils.constants import DATA_DIR


def get_classes() -> list[str]:
    """
    Load and combine class names from disease and pest text files.

    Returns:
        List[str]: Sorted list of unique class names.
    """
    disease_classes_path = DATA_DIR / "disease_classes.txt"
    pest_classes_path = DATA_DIR / "pest_classes.txt"

    classes = []
    for path in [disease_classes_path, pest_classes_path]:
        if path.exists():
            with open(path) as f:
                classes.extend([line.strip() for line in f if line.strip()])
    return sorted(set(classes))

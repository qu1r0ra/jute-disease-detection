import os

from annotator import create_app
from annotator.models import Image, db
from jute_disease.utils import IMAGE_EXTENSIONS, UNLABELED_DIR, get_logger

logger = get_logger(__name__)


def index_unlabeled_images(directory: str) -> None:
    """
    Search a directory for image files and add them to the database if not already
    indexed.

    Args:
        directory (str): Path to the directory containing unlabeled images.
    """
    app = create_app()
    with app.app_context():
        count = 0
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(tuple(IMAGE_EXTENSIONS)):
                    path = os.path.join(root, file)
                    if not Image.query.filter_by(filepath=path).first():
                        img = Image(filename=file, filepath=os.path.abspath(path))
                        db.session.add(img)
                        count += 1
        db.session.commit()
        logger.info(f"Indexed {count} new images into the database.")


if __name__ == "__main__":
    if UNLABELED_DIR.exists():
        index_unlabeled_images(str(UNLABELED_DIR))
    else:
        logger.error(f"Directory {UNLABELED_DIR} not found.")
        logger.info("Create the folder and place your unlabeled images there.")

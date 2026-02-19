from annotator import create_app
from jute_disease.utils import get_logger

logger = get_logger(__name__)
app = create_app()


if __name__ == "__main__":
    logger.info("Starting Jute Disease and Pest Annotation Tool...")
    logger.info("Running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)

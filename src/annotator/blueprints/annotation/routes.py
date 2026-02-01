import os

from flask import Blueprint, render_template, request, send_from_directory

# from annotator.inference import predict_image
from annotator.models import Image, db
from annotator.utils.common import get_classes

annotation_bp = Blueprint("annotation", __name__, template_folder="templates")


@annotation_bp.route("/images/<int:image_id>")
def serve_image(image_id: int):
    """
    Serve a specific image file from the local filesystem.

    Args:
        image_id (int): The database ID of the image.

    Returns:
        Response: The image file response.
    """
    image = Image.query.get_or_404(image_id)
    directory = os.path.dirname(image.filepath)
    filename = os.path.basename(image.filepath)
    return send_from_directory(directory, filename)


@annotation_bp.route("/")
def index() -> str:
    """
    Render the main dashboard with annotation statistics.

    Returns:
        str: Rendered index.html template.
    """
    total = Image.query.count()
    labeled = Image.query.filter_by(is_labeled=True).count()
    return render_template("index.html", total=total, labeled=labeled)


@annotation_bp.route("/annotate")
def annotate() -> str:
    """
    Render the annotation interface for the next unlabeled image.

    Returns:
        str: Rendered annotate.html template or completion message.
    """
    image = Image.query.filter_by(is_labeled=False).first()
    if not image:
        return "All images labeled."

    # NOTE: Trigger model here once trained.
    # predictions = predict_image(image.filepath)

    return render_template("annotate.html", image=image, classes=get_classes())


@annotation_bp.route("/save_label/<int:image_id>", methods=["POST"])
def save_label(image_id: int) -> str:
    """
    Save the user-provided label for an image and serve the next one via HTMX.

    Args:
        image_id (int): The database ID of the image being labeled.

    Returns:
        str: Rendered image_card.html partial or completion message.
    """
    image = Image.query.get_or_404(image_id)
    label = request.form.get("label")

    if label:
        image.label = label
        image.is_labeled = True
        db.session.commit()

    next_image = Image.query.filter_by(is_labeled=False).first()
    if not next_image:
        return "<div class='text-green-500 font-bold'>Annotation completed.</div>"

    return render_template(
        "partials/image_card.html", image=next_image, classes=get_classes()
    )

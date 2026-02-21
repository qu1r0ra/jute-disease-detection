from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(512), unique=True, nullable=False)
    label = db.Column(db.String(100), nullable=True)  # Label given by annotator
    model_prediction = db.Column(db.String(100), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    is_labeled = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    labeled_at = db.Column(db.DateTime, onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        """
        Return a string representation of the Image instance.

        Returns:
            str: Filename and labeling status.
        """
        return (
            f"<Image {self.filename} - {'Labeled' if self.is_labeled else 'Unlabeled'}>"
        )


# TODO: Add Annotation class

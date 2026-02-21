from flask import Flask

from annotator.blueprints.analysis import analysis_bp
from annotator.blueprints.annotation import annotation_bp
from annotator.models import db


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.update(
        SQLALCHEMY_DATABASE_URI="sqlite:///annotations.db",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    db.init_app(app)
    app.register_blueprint(annotation_bp)
    app.register_blueprint(analysis_bp, url_prefix="/analysis")
    with app.app_context():
        db.create_all()
    return app

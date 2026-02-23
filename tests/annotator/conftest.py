import os

import pytest
from flask import Flask

from annotator import create_app
from annotator.models import db


@pytest.fixture
def app() -> Flask:
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    app = create_app()
    app.config.update(
        {
            "TESTING": True,
        }
    )

    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app: Flask):
    return app.test_client()

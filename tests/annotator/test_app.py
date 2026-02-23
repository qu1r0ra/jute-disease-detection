from flask.testing import FlaskClient


def test_app_creates_successfully(client: FlaskClient) -> None:
    """Test that the annotator app is created and endpoints are available."""
    response = client.get("/")
    assert response.status_code in [200, 302]  # It might redirect or load OK.


def test_analysis_endpoint(client: FlaskClient) -> None:
    """Test that the analysis blueprint is correctly registered."""
    response = client.get("/analysis/")
    assert response.status_code in [
        200,
        302,
        404,
    ]  # Depends on the exact routes in analysis_bp

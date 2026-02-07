def test_import_jute_disease():
    import jute_disease

    assert jute_disease is not None


def test_import_classifier():
    from jute_disease.models.jute_classifier import JuteClassifier

    assert JuteClassifier is not None


def test_import_datamodule():
    from jute_disease.data.jute_datamodule import JuteDataModule

    assert JuteDataModule is not None

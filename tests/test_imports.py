def test_import_jute_disease_pest():
    import jute_disease_pest

    assert jute_disease_pest is not None


def test_import_classifier():
    from jute_disease_pest.models.jute_classifier import JuteClassifier

    assert JuteClassifier is not None


def test_import_datamodule():
    from jute_disease_pest.data.jute_datamodule import JuteDataModule

    assert JuteDataModule is not None

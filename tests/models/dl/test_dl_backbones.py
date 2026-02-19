import torch

from jute_disease.models.dl import TimmBackbone


def test_backbone_init():
    # Use a tiny model like resnet18 for fast testing
    model = TimmBackbone(model_name="resnet18", pretrained=False)
    assert model.backbone is not None
    assert hasattr(model, "out_features")
    assert isinstance(model.out_features, int)


def test_backbone_forward():
    model = TimmBackbone(model_name="resnet18", pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)

    assert out.shape[0] == 1
    assert out.shape[1] == model.out_features


def test_backbone_override_features():
    model = TimmBackbone(model_name="resnet18", pretrained=False, out_features=123)
    assert model.out_features == 123

import pytest

torch = pytest.importorskip("torch")

from comas.model import CNNOnlyModel, FusionModel


def test_cnn_only_forward_shape():
    model = CNNOnlyModel(backbone="resnet18", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2,)


def test_fusion_forward_shape():
    text_dim = 384
    model = FusionModel(backbone="resnet18", text_emb_dim=text_dim,
                        pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    t = torch.randn(2, text_dim)
    out = model(x, t)
    assert out.shape == (2,)


def test_cnn_only_accepts_text_emb_kwarg():
    model = CNNOnlyModel(backbone="resnet18", pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    t = torch.zeros(1, 0)
    out = model(x, t)
    assert out.shape == (1,)

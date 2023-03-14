from firecore_oneflow.datasets.fake import FakeDatset
import oneflow as flow


def test_fake():
    fake_ds = FakeDatset(
        image=lambda: flow.rand([2, 3]),
        target=lambda: flow.zeros([], dtype=flow.long)
    )
    assert fake_ds[10]['image'].shape == (2, 3)
    assert fake_ds[10]['target'] == 0

"""
src/models/modules/test_msca.py

تست سریع صحت شکل ورودی/خروجی MSCA (بدون نیاز به GPU یا دیتاست)

نحوه اجرا: از ریشه پروژه (MASW-YOLO-PROJECT) در ترمینال:
    python -m src.models.modules.test_msca
"""

import torch
from .msca import MSCA   # import نسبی چون msca.py در همین پوشه است


def test_msca_shape():
    x = torch.randn(1, 64, 40, 40)
    layer = MSCA(64)
    out = layer(x)
    assert out.shape == x.shape, f"خطا: شکل خروجی {out.shape} با ورودی {x.shape} برابر نیست!"
    print("✅ MSCA درست کار می‌کند. شکل خروجی:", out.shape)


if __name__ == "__main__":
    test_msca_shape()

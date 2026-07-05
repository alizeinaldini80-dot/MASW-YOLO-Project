"""
src/models/modules/test_afpn.py

تست سریع صحت شکل ورودی/خروجی ASFF2 و ASFF3 (بدون نیاز به GPU یا دیتاست)

نحوه اجرا (از ریشه پروژه):
    python -m src.models.modules.test_afpn
"""

import torch
from .afpn import ASFF2, ASFF3


def test_asff2_same_resolution():
    x0 = torch.randn(1, 64, 40, 40)
    x1 = torch.randn(1, 128, 40, 40)
    layer = ASFF2(64, 128, c_out=64, level=0)
    out = layer([x0, x1])
    assert out.shape == (1, 64, 40, 40), f"خطا: شکل خروجی {out.shape}"
    print("✅ ASFF2 (رزولوشن یکسان) درست کار می‌کند. شکل خروجی:", out.shape)


def test_asff2_different_resolution():
    # x0: P3 (رزولوشن بزرگ‌تر) , x1: P4 (رزولوشن کوچک‌تر)
    x0 = torch.randn(1, 64, 80, 80)
    x1 = torch.randn(1, 128, 40, 40)
    layer = ASFF2(64, 128, c_out=64, level=0)  # خروجی در رزولوشن x0 (P3)
    out = layer([x0, x1])
    assert out.shape == (1, 64, 80, 80), f"خطا: شکل خروجی {out.shape}"
    print("✅ ASFF2 (رزولوشن متفاوت، level=0) درست کار می‌کند. شکل خروجی:", out.shape)


def test_asff3_three_levels():
    # شبیه‌سازی سه سطح P3 / P4 / P5 با کانال و رزولوشن متفاوت
    p3 = torch.randn(1, 64, 80, 80)
    p4 = torch.randn(1, 128, 40, 40)
    p5 = torch.randn(1, 64, 20, 20)

    layer_p3 = ASFF3(64, 128, 64, c_out=64, level=0)
    out_p3 = layer_p3([p3, p4, p5])
    assert out_p3.shape == (1, 64, 80, 80)

    layer_p4 = ASFF3(64, 128, 64, c_out=64, level=1)
    out_p4 = layer_p4([p3, p4, p5])
    assert out_p4.shape == (1, 64, 40, 40)

    layer_p5 = ASFF3(64, 128, 64, c_out=64, level=2)
    out_p5 = layer_p5([p3, p4, p5])
    assert out_p5.shape == (1, 64, 20, 20)

    print("✅ ASFF3 برای هر سه سطح (P3/P4/P5) درست کار می‌کند.")


if __name__ == "__main__":
    test_asff2_same_resolution()
    test_asff2_different_resolution()
    test_asff3_three_levels()

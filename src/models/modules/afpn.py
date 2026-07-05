"""
src/models/modules/afpn.py

پیاده‌سازی بخش‌های اصلی Asymptotic Feature Pyramid Network (AFPN)
منطبق بر شکل ۳ مقاله MASW-YOLO:
    - Adaptive Spatial Feature Fusion برای ۲ سطح (ASFF2)
    - Adaptive Spatial Feature Fusion برای ۳ سطح (ASFF3)
    - Basic Block (BB) سبک برای کاهش پارامترها (اختیاری، در صورت نیاز به جایگزینی C2f)

نکته مهم دربارهٔ کانال‌ها:
    برخلاف Conv/C2f که فقط یک ورودی (f=-1 یا یک اندیس) دارند، این دو ماژول
    چند ورودی (f=[i, j] یا f=[i, j, k]) با تعداد کانال‌های متفاوت می‌گیرند.
    parse_model اصلی Ultralytics نمی‌داند چطور کانال این ماژول‌ها را حساب کند،
    بنابراین در custom_model.py یک patch برای این دو کلاس اضافه شده که کانال
    واقعی هر ورودی را از ch[f] می‌خواند و به‌صورت خودکار به سازنده پاس می‌دهد.
    در yaml فقط کافی‌ست بنویسید:
        [[11, 12], 1, ASFF2, [64, 0]]   # args = [c_out, level]
        [[15, 16, 17], 1, ASFF3, [64, 0]]
    یعنی نیازی به نوشتن دستی کانال ورودی‌ها نیست.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resize_to(x: torch.Tensor, size) -> torch.Tensor:
    """تغییر اندازهٔ فضایی x به size؛ برای بزرگ‌نمایی از nearest و برای کوچک‌نمایی از adaptive pooling استفاده می‌شود."""
    if x.shape[-2:] == tuple(size):
        return x
    if x.shape[-2] < size[0]:  # نیاز به upsample
        return F.interpolate(x, size=size, mode="nearest")
    return F.adaptive_avg_pool2d(x, size)  # نیاز به downsample


class ASFF2(nn.Module):
    """
    Adaptive Spatial Feature Fusion برای دو سطح ورودی (معادل بخش سمت راست
    شکل ۳ که با علامت ⊗ و ⊕ نمایش داده شده، اما به‌صورت وزن‌دهی فضایی
    یادگیری‌شونده به‌جای جمع سادهٔ عنصر به عنصر).

    Args (طبق فراخوانی خودکار custom_model.py):
        c1, c2 : تعداد کانال ورودی‌های اول و دوم (خودکار از ch[f] گرفته می‌شود)
        c_out  : تعداد کانال خروجی (از yaml)
        level  : کدام ورودی رزولوشن هدف را تعیین می‌کند (0 یا 1)
    """

    def __init__(self, c1: int, c2: int, c_out: int = None, level: int = 0):
        super().__init__()
        c_out = c_out or c1
        self.level = level

        self.proj0 = nn.Conv2d(c1, c_out, 1) if c1 != c_out else nn.Identity()
        self.proj1 = nn.Conv2d(c2, c_out, 1) if c2 != c_out else nn.Identity()

        # وزن‌های فضایی هر شاخه (Conv1x1 -> 1 کانال) + softmax مشترک
        self.weight0 = nn.Conv2d(c_out, 1, 1)
        self.weight1 = nn.Conv2d(c_out, 1, 1)

        self.conv_out = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x0, x1 = x
        x0 = self.proj0(x0)
        x1 = self.proj1(x1)

        target = x0.shape[-2:] if self.level == 0 else x1.shape[-2:]
        x0 = _resize_to(x0, target)
        x1 = _resize_to(x1, target)

        w = torch.cat([self.weight0(x0), self.weight1(x1)], dim=1)
        w = torch.softmax(w, dim=1)

        fused = x0 * w[:, 0:1] + x1 * w[:, 1:2]
        return self.act(self.bn(self.conv_out(fused)))


class ASFF3(nn.Module):
    """نسخهٔ سه‌ورودی ASFF، برای مرحلهٔ نهایی ادغام (شکل ۱، لایه‌های ۱۸ تا ۲۰ مقاله)."""

    def __init__(self, c1: int, c2: int, c3: int, c_out: int = None, level: int = 0):
        super().__init__()
        c_out = c_out or c1
        self.level = level

        self.proj0 = nn.Conv2d(c1, c_out, 1) if c1 != c_out else nn.Identity()
        self.proj1 = nn.Conv2d(c2, c_out, 1) if c2 != c_out else nn.Identity()
        self.proj2 = nn.Conv2d(c3, c_out, 1) if c3 != c_out else nn.Identity()

        self.weight0 = nn.Conv2d(c_out, 1, 1)
        self.weight1 = nn.Conv2d(c_out, 1, 1)
        self.weight2 = nn.Conv2d(c_out, 1, 1)

        self.conv_out = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x0, x1, x2 = x
        x0 = self.proj0(x0)
        x1 = self.proj1(x1)
        x2 = self.proj2(x2)

        targets = [x0.shape[-2:], x1.shape[-2:], x2.shape[-2:]]
        target = targets[self.level]
        x0 = _resize_to(x0, target)
        x1 = _resize_to(x1, target)
        x2 = _resize_to(x2, target)

        w = torch.cat([self.weight0(x0), self.weight1(x1), self.weight2(x2)], dim=1)
        w = torch.softmax(w, dim=1)

        fused = x0 * w[:, 0:1] + x1 * w[:, 1:2] + x2 * w[:, 2:3]
        return self.act(self.bn(self.conv_out(fused)))


class BasicBlock(nn.Module):
    """
    Basic Block (BB) سبک‌وزن که در مقاله برای کاهش پارامترها در AFPN معرفی شده.
    اختیاری است؛ در yaml پیش‌فرض ما به‌جای آن از C2f موجود در Ultralytics استفاده
    شده (چون از نظر عملکردی مشابه است)، اما اگر خواستی دقیقاً طبق مقاله عمل کنی
    می‌توانی لایه‌های C2f بعد از هر ASFF را با BasicBlock جایگزین کنی.
    """

    def __init__(self, c1: int, c2: int = None, shortcut: bool = True):
        super().__init__()
        c2 = c2 or c1
        self.cv1 = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.cv2 = nn.Conv2d(c2, c2, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.act(self.bn1(self.cv1(x)))
        y = self.bn2(self.cv2(y))
        return self.act(x + y) if self.add else self.act(y)

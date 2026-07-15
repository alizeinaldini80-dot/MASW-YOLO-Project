"""
src/models/modules/afpn.py  (v2 - بازنویسی کامل)

پیاده‌سازی وفادار به شکل ۳ مقاله: Asymptotic Feature Pyramid Network (AFPN).

طبق متن مقاله سه رکن اصلی AFPN عبارت‌اند از:
    - CB  (Convolutional Block)   -> اینجا از طریق کلاس Conv (=CBS: Conv-BN-SiLU)
    - BB  (Basic Block)           -> کلاس BasicBlock
    - ASFF (Adaptive Spatial Feature Fusion) -> کلاس‌های ASFF2 / ASFF3 / ASFF4

و ساختار fusion باید «تدریجی/مجانبی» (asymptotic) باشد، دقیقاً طبق متن:
    مرحله ۱: C2 و C3 با هم ترکیب می‌شوند (ASFF2)
    مرحله ۲: نتیجهٔ مرحلهٔ ۱ با C4 ترکیب می‌شود (ASFF3)
    مرحله ۳: نتیجهٔ مرحلهٔ ۲ با C5 ترکیب می‌شود (ASFF4)

هر مرحله بعد از fusion یک BasicBlock برای پالایش ویژگی (طبق متن: "lightens the
model and minimizes the number of parameters... perform feature extraction
more efficiently") دارد.

خروجی نهایی چهار نقشهٔ ویژگی هم‌کانال [P2, P3, P4, P5] است — دقیقاً چهار
سطحی که در شکل ۳ به "Predict" ختم می‌شوند.

نکته دربارهٔ استفاده در YOLO head: از آنجا که این ماژول یک لیست ۴تایی
برمی‌گرداند (نه یک تنسور)، یک ماژول کمکی `Index` هم اضافه شده که هر یک از
چهار خروجی را برای مصرف در لایه‌های بعدی (Concat/C2f/Detect) جدا می‌کند —
این الگوی رایج در پیاده‌سازی‌های سفارشی ultralytics برای ماژول‌های
چندخروجی است.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# CB : Convolutional Block  (= CBS : Conv -> BatchNorm -> SiLU)
# --------------------------------------------------------------------------- #
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """CBS: Conv2d -> BatchNorm2d -> SiLU (متن مقاله: "the CBS module's job is
    to slice and extend the input feature maps in channel dimensions")."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# --------------------------------------------------------------------------- #
# BB : Basic Block  (رفرش/پالایش ویژگی بعد از هر مرحلهٔ fusion)
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    """بلوک residual سبک: دو کانولوشن ۳×۳ + جمع باقی‌مانده.
    طبق متن مقاله: "BasicBlock module lightens the model and minimizes the
    number of parameters, allowing it to perform feature extraction more
    efficiently"."""

    def __init__(self, c1, c2=None):
        super().__init__()
        c2 = c2 or c1
        self.cv1 = Conv(c1, c2, 3, 1)
        self.cv2 = Conv(c2, c2, 3, 1, act=False)
        self.act = nn.SiLU()
        self.add = c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return self.act(y + x) if self.add else self.act(y)


# --------------------------------------------------------------------------- #
# کمکی: هم‌رزولوشن‌سازی نقشه‌های ویژگی قبل از fusion
# --------------------------------------------------------------------------- #
def _resize_to(x, size):
    if x.shape[-2:] == size:
        return x
    if x.shape[-2] > size[0]:
        # downsample: average pooling (شبیه استاندارد پیاده‌سازی ASFF اصلی)
        return F.adaptive_avg_pool2d(x, size)
    # upsample
    return F.interpolate(x, size=size, mode="nearest")


# --------------------------------------------------------------------------- #
# ASFF : Adaptive Spatial Feature Fusion  (نسخه‌های ۲، ۳ و ۴ ورودی)
# --------------------------------------------------------------------------- #
class _ASFFBase(nn.Module):
    """پایهٔ مشترک ASFFn: هر سطح یک نقشهٔ وزن تک‌کاناله می‌سازد، روی سطوح
    softmax می‌شود و ترکیب وزن‌دار انجام می‌شود، سپس یک Conv3x3 خروجی را
    یکپارچه می‌کند."""

    def __init__(self, n_levels, c1, level=0):
        super().__init__()
        self.n_levels = n_levels
        self.level = level
        self.weights = nn.ModuleList([nn.Conv2d(c1, 1, 1) for _ in range(n_levels)])
        self.conv_out = Conv(c1, c1, 3, 1)

    def forward(self, xs):
        assert len(xs) == self.n_levels
        target = xs[self.level].shape[-2:]
        xs = [_resize_to(xi, target) for xi in xs]
        w = torch.cat([wf(xi) for wf, xi in zip(self.weights, xs)], dim=1)
        w = torch.softmax(w, dim=1)
        fused = sum(xi * w[:, i : i + 1] for i, xi in enumerate(xs))
        return self.conv_out(fused)


class ASFF2(_ASFFBase):
    """fusion دو سطحی — طبق شکل ۳: مرحلهٔ اول، فقط C2 و C3."""

    def __init__(self, c0, c1, c_out, level=0):
        # c0, c1 توسط parsing.py پاس داده می‌شوند اما چون همهٔ ورودی‌ها قبلاً
        # روی c_out پروجکت شده‌اند (conv های proj در AFPN)، این دو آرگومان
        # فقط برای سازگاری با امضای parse_model نگه داشته شده‌اند.
        super().__init__(n_levels=2, c1=c_out, level=level)


class ASFF3(_ASFFBase):
    """fusion سه سطحی — طبق شکل ۳: مرحلهٔ دوم، بعد از اضافه‌شدن C4."""

    def __init__(self, c0, c1, c2, c_out, level=0):
        super().__init__(n_levels=3, c1=c_out, level=level)


class ASFF4(_ASFFBase):
    """fusion چهار سطحی — طبق شکل ۳: مرحلهٔ سوم/نهایی، بعد از اضافه‌شدن C5."""

    def __init__(self, c0, c1, c2, c3, c_out, level=0):
        super().__init__(n_levels=4, c1=c_out, level=level)


# --------------------------------------------------------------------------- #
# AFPN : اورکستریشن کامل fusion تدریجی (شکل ۳)
# --------------------------------------------------------------------------- #
class AFPN(nn.Module):
    """
    ورودی : لیست ۴ نقشهٔ ویژگی بک‌بون [C2, C3, C4, C5] با کانال‌های دلخواه.
    خروجی : لیست ۴ نقشهٔ ویژگی هم‌کانال [P2, P3, P4, P5] با عرض `width`.

    Progressive / asymptotic fusion طبق متن مقاله:
        Stage 1 : C2, C3                     -> ASFF2 -> BasicBlock  (سطح ۲ خروجی)
        Stage 2 : Stage1_out(2) , C4          -> ASFF3 -> BasicBlock  (سطح ۳ خروجی)
        Stage 3 : Stage2_out(3) , C5          -> ASFF4 -> BasicBlock  (سطح ۴ خروجی نهایی)
    """

    def __init__(self, width, c2_in, c3_in, c4_in, c5_in):
        super().__init__()
        w = width

        # --- CB: پروجکشن اولیهٔ همهٔ سطوح به عرض مشترک w ---
        self.proj2 = Conv(c2_in, w, 1, 1)
        self.proj3 = Conv(c3_in, w, 1, 1)
        self.proj4 = Conv(c4_in, w, 1, 1)
        self.proj5 = Conv(c5_in, w, 1, 1)

        # --- Stage 1: C2 + C3 ---
        self.asff2_l0 = ASFF2(w, w, w, level=0)
        self.asff2_l1 = ASFF2(w, w, w, level=1)
        self.bb1_0 = BasicBlock(w)
        self.bb1_1 = BasicBlock(w)

        # --- Stage 2: (+C4) ---
        self.asff3_l0 = ASFF3(w, w, w, w, level=0)
        self.asff3_l1 = ASFF3(w, w, w, w, level=1)
        self.asff3_l2 = ASFF3(w, w, w, w, level=2)
        self.bb2_0 = BasicBlock(w)
        self.bb2_1 = BasicBlock(w)
        self.bb2_2 = BasicBlock(w)

        # --- Stage 3: (+C5) ---
        self.asff4_l0 = ASFF4(w, w, w, w, w, level=0)
        self.asff4_l1 = ASFF4(w, w, w, w, w, level=1)
        self.asff4_l2 = ASFF4(w, w, w, w, w, level=2)
        self.asff4_l3 = ASFF4(w, w, w, w, w, level=3)
        self.bb3_0 = BasicBlock(w)
        self.bb3_1 = BasicBlock(w)
        self.bb3_2 = BasicBlock(w)
        self.bb3_3 = BasicBlock(w)

    def forward(self, x):
        c2, c3, c4, c5 = x
        c2 = self.proj2(c2)
        c3 = self.proj3(c3)
        c4 = self.proj4(c4)
        c5 = self.proj5(c5)

        # ---- Stage 1: "the semantic information of C2 and C3 is first integrated" ----
        s1 = [c2, c3]
        p2 = self.bb1_0(self.asff2_l0(s1))
        p3 = self.bb1_1(self.asff2_l1(s1))

        # ---- Stage 2: "the semantic information of C3 and C4 is amalgamated" ----
        s2 = [p2, p3, c4]
        p2 = self.bb2_0(self.asff3_l0(s2))
        p3 = self.bb2_1(self.asff3_l1(s2))
        p4 = self.bb2_2(self.asff3_l2(s2))

        # ---- Stage 3: نهایتاً C5 هم اضافه می‌شود ----
        s3 = [p2, p3, p4, c5]
        p2 = self.bb3_0(self.asff4_l0(s3))
        p3 = self.bb3_1(self.asff4_l1(s3))
        p4 = self.bb3_2(self.asff4_l2(s3))
        p5 = self.bb3_3(self.asff4_l3(s3))

        return [p2, p3, p4, p5]


# --------------------------------------------------------------------------- #
# Index : استخراج یک خروجی از یک لیست چندتایی (برای استفاده در YAML بعد از AFPN)
# --------------------------------------------------------------------------- #
class Index(nn.Module):
    """چون AFPN یک لیست ۴تایی برمی‌گرداند (نه یک تنسور تکی)، این ماژول
    عنصر i-اُم آن لیست را برای مصرف در لایه‌های بعدی (Concat/C2f/Detect)
    جدا می‌کند. الگوی رایج در پیاده‌سازی‌های سفارشی ultralytics برای
    ماژول‌های چندخروجی."""

    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, x):
        return x[self.idx]

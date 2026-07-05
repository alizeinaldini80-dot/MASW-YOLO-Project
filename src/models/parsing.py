"""
src/models/parsing.py

نسخهٔ گسترش‌یافتهٔ parse_model اصلی Ultralytics. این فایل تقریباً عیناً
همان منطق تابع ultralytics.nn.tasks.parse_model (نسخهٔ نصب‌شده هنگام
نوشتن این کد) است، با این تفاوت که سه شاخهٔ اضافه برای MSCA، ASFF2 و
ASFF3 دارد. تمام ماژول‌های استاندارد دیگر (Conv, C2f, SPPF, Concat,
Detect, و ...) دقیقاً همان مسیر اصلی Ultralytics را طی می‌کنند.

⚠️ توجه به نسخه: اگر بعد از آپدیت ultralytics در کولب باز هم خطای
عجیب گرفتید (مخصوصاً دربارهٔ آرگومان‌های Detect/reg_max/end2end)،
این دستور را در کولب اجرا کنید و خروجی‌اش را با این فایل مقایسه کنید:
    import inspect, ultralytics.nn.tasks as t
    print(inspect.getsource(t.parse_model))
"""

import ast
import contextlib

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv, C2f, SPPF, Concat, Detect
from ultralytics.utils import LOGGER
from ultralytics.utils.ops import make_divisible

from .modules.msca import MSCA
from .modules.afpn import ASFF2, ASFF3

_MULTI_INPUT_MODULES = {"ASFF2": ASFF2, "ASFF3": ASFF3}
_CUSTOM_MODULES = {"MSCA": MSCA, "ASFF2": ASFF2, "ASFF3": ASFF3}


def uses_custom_multi_input(d: dict) -> bool:
    """آیا این کانفیگ yaml حداقل یک لایه ASFF2/ASFF3 دارد؟"""
    layer_defs = d.get("backbone", []) + d.get("head", [])
    return any(layer[2] in _MULTI_INPUT_MODULES for layer in layer_defs)


def parse_model_with_custom_modules(d: dict, ch, verbose: bool = True):
    """
    جایگزین parse_model اصلی، فقط برای yaml هایی که ASFF2/ASFF3 دارند
    (توسط wrapper در __init__.py قبل از فراخوانی چک می‌شود).
    برای ماژول‌های غیر سفارشی، همان مسیر ultralytics.nn.tasks.parse_model
    استفاده می‌شود تا رفتار همیشه با نسخهٔ نصب‌شده هماهنگ بماند.
    """
    import ultralytics.nn.tasks as ultra_tasks

    legacy = True
    max_channels = float("inf")
    nc = d.get("nc")
    scales = d.get("scales")
    end2end = d.get("end2end")
    reg_max = d.get("reg_max", 16)
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    scale = d.get("scale")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
        depth, width, max_channels = scales[scale]

    ch = [ch]
    layers, save, c2 = [], [], ch[-1]

    for i, (f, n, m_name, args) in enumerate(d["backbone"] + d["head"]):
        args = list(args)

        # --- حل نام ماژول به کلاس واقعی ---
        if isinstance(m_name, str):
            if m_name.startswith("nn."):
                m = getattr(nn, m_name[3:])
            elif m_name in _CUSTOM_MODULES:
                m = _CUSTOM_MODULES[m_name]
            else:
                # هر ماژول استاندارد دیگر Ultralytics (Conv, C2f, SPPF, Detect, ...)
                m = getattr(ultra_tasks, m_name, None)
                if m is None:
                    raise ValueError(f"ماژول ناشناخته در yaml: {m_name}")
        else:
            m = m_name

        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n

        # ============== شاخه‌های سفارشی ما ==============
        if m in (ASFF2, ASFF3):
            c_out_raw, level = args[0], (args[1] if len(args) > 1 else 0)
            c_out = make_divisible(min(c_out_raw, max_channels) * width, 8)
            in_channels = [ch[x] for x in f]
            args = [*in_channels, c_out, level]
            c2 = c_out

        elif m is MSCA:
            c1 = ch[f]
            args = [c1]
            c2 = c1

        # ============== از اینجا به بعد دقیقاً منطق اصلی Ultralytics ==============
        elif m in (Conv, C2f, SPPF):
            c1, c2_ = ch[f], args[0]
            if c2_ != nc:
                c2_ = make_divisible(min(c2_, max_channels) * width, 8)
            args = [c1, c2_, *args[1:]]
            if m is C2f:
                args.insert(2, n)
                n = 1
            c2 = c2_

        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
            c2 = ch[f]

        elif m is Concat:
            c2 = sum(ch[x] for x in f)

        elif m is Detect:
            # طبق نسخهٔ نصب‌شدهٔ فعلی Ultralytics: (nc, reg_max, end2end, ch)
            args.extend([reg_max, end2end, [ch[x] for x in f]])
            m.legacy = legacy

        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        m_.i, m_.f, m_.type = i, f, t
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}  {t:<40}{str(args):<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)

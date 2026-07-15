
#src/models/parsing.py


import ast
import contextlib

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv, C2f, SPPF, Concat, Detect
from ultralytics.utils import LOGGER
from ultralytics.utils.ops import make_divisible

from .modules.msca import MSCA
from .modules.afpn import ASFF2, ASFF3, ASFF4, AFPN, Index

_MULTI_INPUT_MODULES = {"ASFF2": ASFF2, "ASFF3": ASFF3, "ASFF4": ASFF4, "AFPN": AFPN}
_CUSTOM_MODULES = {
    "MSCA": MSCA,
    "ASFF2": ASFF2,
    "ASFF3": ASFF3,
    "ASFF4": ASFF4,
    "AFPN": AFPN,
    "Index": Index,
}


def uses_custom_multi_input(d: dict) -> bool:
    """آیا این کانفیگ yaml حداقل یک لایه ASFF2/ASFF3/ASFF4/AFPN دارد؟"""
    layer_defs = d.get("backbone", []) + d.get("head", [])
    return any(layer[2] in _MULTI_INPUT_MODULES or layer[2] == "Index" for layer in layer_defs)


def parse_model_with_custom_modules(d: dict, ch, verbose: bool = True):
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

        if isinstance(m_name, str):
            if m_name.startswith("nn."):
                m = getattr(nn, m_name[3:])
            elif m_name in _CUSTOM_MODULES:
                m = _CUSTOM_MODULES[m_name]
            else:
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

        if m in (ASFF2, ASFF3, ASFF4):
            c_out_raw, level = args[0], (args[1] if len(args) > 1 else 0)
            c_out = make_divisible(min(c_out_raw, max_channels) * width, 8)
            in_channels = [ch[x] for x in f]
            args = [*in_channels, c_out, level]
            c2 = c_out

        elif m is AFPN:
            width_raw = args[0]
            w = make_divisible(min(width_raw, max_channels) * width, 8)
            in_channels = [ch[x] for x in f] 
            args = [w, *in_channels]
            c2 = w 

        elif m is Index:
            idx = args[0]
            args = [idx]
            c2 = ch[f] 
        elif m is MSCA:
            c1 = ch[f]
            args = [c1]
            c2 = c1

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

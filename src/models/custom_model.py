"""
Custom MASW-YOLO Detection Model.

Defines `MASWDetectionModel` which inherits from
`ultralytics.nn.tasks.DetectionModel` and registers the custom MSCA and
AFPN (ASFF2 / ASFF3) layers so that a YAML config file can reference them
by name.
"""

import torch
import torch.nn as nn

from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import (
    Conv,
    C2f,
    SPPF,
    Detect,
    Concat,
)

from .modules.msca import MSCA
from .modules.afpn import ASFF2, ASFF3, BasicBlock

# نام‌هایی که parse_model باید آن‌ها را به‌صورت «چند ورودی با کانال متفاوت» بشناسد.
# برای این ماژول‌ها در yaml فقط [c_out, level] نوشته می‌شود، کانال ورودی‌ها
# به‌صورت خودکار از ch[f] خوانده می‌شود.
_MULTI_INPUT_MODULES = {"ASFF2": ASFF2, "ASFF3": ASFF3}


def custom_parse_model(cfg: dict, ch: list, verbose: bool = True):
    """
    Custom model-parsing helper that maps string layer names (including
    ``'MSCA'``, ``'ASFF2'`` and ``'ASFF3'``) to their corresponding
    ``nn.Module`` classes, then calls ``parse_model`` on the
    ultralytics-style YAML configuration.

    برخلاف Conv/C2f (تک‌ورودی)، ماژول‌های ASFF2/ASFF3 چند ورودی با تعداد
    کانال متفاوت دارند. تابع اصلی parse_model در Ultralytics نمی‌داند این
    کانال‌ها را چطور محاسبه کند (چون فقط برای Concat و Detect حالت خاص
    نوشته شده)، بنابراین اینجا خودمان یک نسخهٔ patch‌شده از منطق آن را
    پیاده می‌کنیم.

    Args:
        cfg (dict): Model configuration dictionary (from parsed YAML).
        ch (list):  List of input channels for each layer.
        verbose (bool): Whether to print layer information.

    Returns:
        (nn.ModuleList, list): مطابق خروجی استاندارد parse_model.
    """
    import ultralytics.nn.modules as ultra_modules
    import ultralytics.nn.tasks as ultra_tasks

    # ثبت ماژول‌های سفارشی در namespace اصلی Ultralytics
    ultra_modules.MSCA = MSCA
    ultra_modules.ASFF2 = ASFF2
    ultra_modules.ASFF3 = ASFF3
    ultra_modules.BasicBlock = BasicBlock

    if hasattr(ultra_tasks, "modules"):
        ultra_tasks.modules["MSCA"] = MSCA
        ultra_tasks.modules["ASFF2"] = ASFF2
        ultra_tasks.modules["ASFF3"] = ASFF3
        ultra_tasks.modules["BasicBlock"] = BasicBlock

    # اگر yaml شامل ASFF2/ASFF3 نباشد (مثلاً exp1_msca.yaml)، نیازی به
    # patch اضافه نیست و مستقیم به parse_model اصلی واگذار می‌کنیم.
    layer_defs = cfg.get("backbone", []) + cfg.get("head", [])
    uses_multi_input = any(layer[2] in _MULTI_INPUT_MODULES for layer in layer_defs)

    if not uses_multi_input:
        model, save = ultra_tasks.parse_model(cfg, ch, verbose)
        return model, save

    return _parse_model_with_asff(cfg, ch, verbose)


def _parse_model_with_asff(d: dict, ch: list, verbose: bool = True):
    """
    نسخهٔ گسترش‌یافتهٔ منطق parse_model اصلی Ultralytics، فقط برای مدیریت
    ASFF2/ASFF3. برای سایر ماژول‌ها (Conv, C2f, SPPF, Concat, Detect, ...)
    دقیقاً از توابع/کلاس‌های خود Ultralytics استفاده می‌شود.

    ⚠️ توجه: این تابع بر اساس ساختار عمومی parse_model در نسخه‌های ۸.x
    Ultralytics نوشته شده. اگر نسخهٔ نصب‌شدهٔ شما تفاوت زیادی داشت و خطای
    غیرمنتظره گرفتید، ساختار backbone/head خود را ساده نگه دارید یا این
    تابع را با خروجی ``inspect.getsource(ultralytics.nn.tasks.parse_model)``
    مقایسه کنید.
    """
    import ast
    import contextlib
    import ultralytics.nn.tasks as ultra_tasks
    from ultralytics.utils import LOGGER
    from ultralytics.utils.ops import make_divisible

    nc = d.get("nc", 80)
    scales = d.get("scales")
    depth, width, max_channels = 1.0, 1.0, float("inf")
    if scales:
        scale = d.get("scale") or next(iter(scales.keys()))
        depth, width, max_channels = scales[scale]

    ch = [ch] if isinstance(ch, int) else list(ch)
    layers, save, c2 = [], [], ch[-1]

    for i, (f, n, m_name, args) in enumerate(d["backbone"] + d["head"]):
        args = list(args)

        # حل نام ماژول به کلاس واقعی
        if isinstance(m_name, str):
            if m_name.startswith("nn."):
                m = getattr(nn, m_name[3:])
            elif m_name in _MULTI_INPUT_MODULES:
                m = _MULTI_INPUT_MODULES[m_name]
            else:
                m = getattr(ultra_tasks, m_name, None) or getattr(nn, m_name, None)
                if m is None:
                    raise ValueError(f"ماژول ناشناخته در yaml: {m_name}")
        else:
            m = m_name

        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n

        if m is _MULTI_INPUT_MODULES.get("ASFF2") or m is _MULTI_INPUT_MODULES.get("ASFF3"):
            # args طبق yaml: [c_out, level]
            c_out_raw, level = args[0], (args[1] if len(args) > 1 else 0)
            c_out = make_divisible(min(c_out_raw, max_channels) * width, 8)
            in_channels = [ch[x] for x in f]
            args = [*in_channels, c_out, level]
            c2 = c_out

        elif m in (Conv, C2f, SPPF):
            c1, c2_ = ch[f], args[0]
            if c2_ != nc:
                c2_ = make_divisible(min(c2_, max_channels) * width, 8)
            args = [c1, c2_, *args[1:]]
            if m is C2f:
                args.insert(2, n)
                n = 1
            c2 = c2_

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
            c2 = ch[f]

        elif m is Concat:
            c2 = sum(ch[x] for x in f)

        elif m is Detect:
            args.append([ch[x] for x in f])

        elif m is MSCA:
            c1 = ch[f]
            args = [c1]
            c2 = c1

        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        m_.i, m_.f, m_.type = i, f, t
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}  {t:<40}{str(args):<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)


class MASWDetectionModel(DetectionModel):
    """
    MASW-YOLO Detection Model.

    Extends ``ultralytics.nn.tasks.DetectionModel`` by registering the
    custom ``MSCA``, ``ASFF2`` و ``ASFF3`` layers so they can be used
    inside a YOLO-style model YAML configuration.

    Usage:
        model = MASWDetectionModel("path/to/masw-yolo-afpn.yaml", nc=10)
    """

    def __init__(
        self,
        cfg: str = "yolov8n.yaml",
        ch: int = 3,
        nc: int = 80,
        verbose: bool = True,
    ):
        import yaml  # lazy import

        with open(cfg, encoding="ascii", errors="ignore") as f:
            self.yaml = yaml.safe_load(f)

        ch = [ch]
        self.model, self.save = custom_parse_model(self.yaml, ch, verbose)
        self.names = {i: f"class{i}" for i in range(nc)}
        self.inplace = self.yaml.get("inplace", True)

        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256
            m.inplace = self.inplace

            def _forward(x):
                return self.forward(x)

            m.stride = torch.tensor(
                [s / x.shape[-2] for x in _forward(torch.zeros(1, ch[0], s, s))]
            )
            self.stride = m.stride
            m.nc = nc
            m.names = self.names
        else:
            self.stride = torch.Tensor([32])

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif t is nn.BatchNorm2d:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif t is Detect:
                m._initialize_biases()


#src/models/__init__.py


import ultralytics.nn.tasks as tasks

from .modules.msca import MSCA
from .modules.afpn import ASFF2, ASFF3, ASFF4, AFPN, BasicBlock, Index
from .parsing import parse_model_with_custom_modules, uses_custom_multi_input

tasks.MSCA = MSCA
tasks.ASFF2 = ASFF2
tasks.ASFF3 = ASFF3
tasks.ASFF4 = ASFF4
tasks.AFPN = AFPN
tasks.BasicBlock = BasicBlock
tasks.Index = Index


_original_parse_model = tasks.parse_model


def _patched_parse_model(d, ch, verbose=True):
    if uses_custom_multi_input(d):
        return parse_model_with_custom_modules(d, ch, verbose=verbose)
    return _original_parse_model(d, ch, verbose=verbose)


tasks.parse_model = _patched_parse_model

__all__ = ["MSCA", "ASFF2", "ASFF3", "ASFF4", "AFPN", "BasicBlock", "Index"]

print("✅ ماژول‌های سفارشی MASW-YOLO ثبت شدند: MSCA, ASFF2, ASFF3, ASFF4, AFPN, BasicBlock, Index")
print("✅ parse_model برای پشتیبانی از ورودی/خروجی چندگانه AFPN patch شد")

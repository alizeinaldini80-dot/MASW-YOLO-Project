"""
src/models/__init__.py

این فایل دو کار انجام می‌دهد:
۱) ثبت کلاس‌های سفارشی (MSCA, ASFF2, ASFF3, ASFF4, AFPN, BasicBlock, Index)
   در namespace داخلی Ultralytics تا yaml بتواند اسم آن‌ها را resolve کند.
۲) monkey-patch کردن خودِ تابع ultralytics.nn.tasks.parse_model، چون تابع
   اصلی نمی‌داند کانال ورودی‌های چندگانهٔ ASFF2/ASFF3/ASFF4/AFPN (که هرکدام
   کانال متفاوت دارند) یا خروجی چندتایی AFPN را چطور مدیریت کند. این patch
   فقط وقتی فعال می‌شود که yaml واقعاً شامل یکی از این ماژول‌ها باشد؛ برای
   yaml های معمولی (baseline، msca-only) هیچ تغییری در رفتار قبلی ایجاد
   نمی‌شود.

نکته: چون این ثبت‌نام یک‌بار در سطح ماژول انجام می‌شود، هم مسیر
`YOLO(cfg['model_yaml'])` (که در train.py استفاده می‌کنید) و هم مسیر
`MASWDetectionModel(...)` به‌طور یکسان از آن بهره می‌برند.
"""

import ultralytics.nn.tasks as tasks

from .modules.msca import MSCA
from .modules.afpn import ASFF2, ASFF3, ASFF4, AFPN, BasicBlock, Index
from .parsing import parse_model_with_custom_modules, uses_custom_multi_input

# --- ۱) ثبت نام کلاس‌ها ---
tasks.MSCA = MSCA
tasks.ASFF2 = ASFF2
tasks.ASFF3 = ASFF3
tasks.ASFF4 = ASFF4
tasks.AFPN = AFPN
tasks.BasicBlock = BasicBlock
tasks.Index = Index

# --- ۲) monkey-patch تابع parse_model ---
_original_parse_model = tasks.parse_model


def _patched_parse_model(d, ch, verbose=True):
    if uses_custom_multi_input(d):
        return parse_model_with_custom_modules(d, ch, verbose=verbose)
    return _original_parse_model(d, ch, verbose=verbose)


tasks.parse_model = _patched_parse_model

__all__ = ["MSCA", "ASFF2", "ASFF3", "ASFF4", "AFPN", "BasicBlock", "Index"]

print("✅ ماژول‌های سفارشی MASW-YOLO ثبت شدند: MSCA, ASFF2, ASFF3, ASFF4, AFPN, BasicBlock, Index")
print("✅ parse_model برای پشتیبانی از ورودی/خروجی چندگانه AFPN patch شد")

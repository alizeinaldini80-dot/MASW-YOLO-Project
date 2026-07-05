"""
src/models/__init__.py

این فایل ماژول‌های سفارشی MASW-YOLO (MSCA و AFPN) را در فضای نام
داخلی Ultralytics ثبت می‌کند تا فایل‌های معماری .yaml بتوانند این نام‌ها را بشناسند.

نکته: این ثبت‌نام یک بار در سطح ماژول انجام می‌شود و برای هر مسیری که مدل را
می‌سازد کافی است (چه از طریق MASWDetectionModel در custom_model.py، چه از
طریق YOLO() مستقیم اولترالیتیکس با yaml سفارشی).
"""

import ultralytics.nn.tasks as tasks
from .modules.msca import MSCA
from .modules.afpn import ASFF2, ASFF3, BasicBlock

tasks.MSCA = MSCA
tasks.ASFF2 = ASFF2
tasks.ASFF3 = ASFF3
tasks.BasicBlock = BasicBlock

# توجه: در این پیاده‌سازی «CB» یک کلاس جداگانه نیست؛ نقش آن (یکسان‌سازی کانال
# قبل از ASFF) با لایه‌های استاندارد Conv در خودِ yaml انجام می‌شود
# (نگاه کنید به لایه‌های 11، 12، 17 در masw-yolo-afpn.yaml).

__all__ = ["MSCA", "ASFF2", "ASFF3", "BasicBlock"]

print("✅ ماژول‌های سفارشی MASW-YOLO ثبت شدند: MSCA, ASFF2, ASFF3, BasicBlock")

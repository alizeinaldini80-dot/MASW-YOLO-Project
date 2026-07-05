from .modules.msca import MSCA
from .modules.afpn import ASFF2, ASFF3, BasicBlock
import ultralytics.nn.tasks as tasks

# ثبت در فضای نام Ultralytics
tasks.MSCA = MSCA
tasks.ASFF2 = ASFF2
tasks.ASFF3 = ASFF3
tasks.BasicBlock = BasicBlock

# همچنین می‌توانید به ماژول‌های خود اشاره کنید
__all__ = ["MSCA", "ASFF2", "ASFF3", "BasicBlock"]

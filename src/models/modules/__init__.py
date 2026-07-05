# src/models/modules/__init__.py
from .msca import MSCA
from .afpn import ASFF2, ASFF3, BasicBlock
import ultralytics.nn.tasks as tasks

tasks.MSCA = MSCA
tasks.ASFF2 = ASFF2
tasks.ASFF3 = ASFF3
tasks.BasicBlock = BasicBlock

__all__ = ["MSCA", "ASFF2", "ASFF3", "BasicBlock"]

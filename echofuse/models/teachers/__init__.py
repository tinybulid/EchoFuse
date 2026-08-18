"""Teacher architectures preserved from the previous package.

Only the import registry was standardized so the existing architecture files
can be built reliably from this package.
"""

from .Lenv1 import DSFAmirTeacher
from .LenV2 import DSFMehdy2Teacher
from .LenV2Prime import DSFMehdyTeacher
from .cp_mobile import CPMobileTeacher
from .cp_resnet import CPResNetTeacher
from .registry import available_external_teachers, build_external_teacher

__all__ = [
    "DSFAmirTeacher",
    "DSFMehdy2Teacher",
    "DSFMehdyTeacher",
    "CPMobileTeacher",
    "CPResNetTeacher",
    "available_external_teachers",
    "build_external_teacher",
]

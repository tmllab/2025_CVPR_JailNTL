"""
This package contains modules related to objective functions, optimizations, and network architectures.
This file defines the interface for creating models.
"""

from .base_model import BaseModel

def create_model(opt, mode, task_name, ntl_model=None):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    if mode == 'test':
        from .test_model import TestModel
        instance = TestModel(opt, task_name)
    elif mode == 'train':
        from .disguising_model import DisguisingModel
        instance = DisguisingModel(opt, task_name, ntl_model)
    print("model [%s] was created" % type(instance).__name__)
    return instance

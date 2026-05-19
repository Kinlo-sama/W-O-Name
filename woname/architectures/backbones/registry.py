# backbone/registry.py

from typing import Dict, Type

from .base import BackboneBase
from .configs import BackboneConfig

BACKBONE_REGISTRY: Dict[str, Type[BackboneConfig]] = {}

def register_backbone(name: str):
    """
    Decorator to register backbone classes.
    """

    def decorator(cls):
        if name in BACKBONE_REGISTRY:
           raise ValueError(
               f"Backbone '{name}' is already registered."
           ) 
        if not issubclass(cls, BackboneBase):
            raise TypeError(
                f"{cls.__name__} must inherit from BackboneBase"
            )
        
        BACKBONE_REGISTRY[name] = cls

        return cls 
    return decorator

def build_backbone(cfg: BackboneConfig) -> BackboneBase:
    """
    Build a backbone instance from config
    """

    backbone_type = cfg.type

    if backbone_type not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknow backbone '{backbone_type}'. "
            f"Available backbones: {list(BACKBONE_REGISTRY.keys())}"
        )
    
    backbone_class = BACKBONE_REGISTRY[backbone_type]

    return backbone_class(cfg)
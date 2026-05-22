from typing import Any, Dict

class Registry:
    """
    Generic registry system.
    """

    def __init__(self, name: str):
        self.name = name
        self._modules : Dict[str, Any] = {}

    def register(self, name: str):
        """
        Register a module.
        """

        def decorator(cls):
            if name in self._modules:
                raise ValueError(
                    f"'{name}' is already registered"
                    f"in registry '{self.name}'"
                )
            self._modules[name] = cls

            return cls 
        return decorator
    
    def get(self, name: str):
        if name not in self._modules:
            raise KeyError(
                f"'{name}' is not registered"
                f"in registry '{self.name}'"
            )
        return self._modules[name]
    
    def build(self, cfg, *args, **kwargs):

        module_type = cfg.type

        module_class = self._modules[module_type]

        return module_class(
            cfg,
            *args,
            **kwargs
        )
    
    def list_modules(self):
        return list(self._modules.keys())

    def __contains__(self, name):
        return name in self._modules
    
    def __repr__(self):
        return (
            f"Registry(name={self.name}), "
            f"items={list(self._modules.keys())}"
        )
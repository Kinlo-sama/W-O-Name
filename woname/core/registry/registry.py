from typing import Any, Dict, Optional

class Registry:
    """
    Generic registry system.
    """

    def __init__(self, name: str):
        self.name = name
        self._modules : Dict[str, Any] = {}
        self._configs : Dict[str, Any] = {}

    def register(self, name: str, config: Optional[Any] = None):
        """
        Register a module.
        """

        def decorator(cls):
            if name in self._modules:
                raise ValueError(
                    f"'{name}' is already registered "
                    f"in registry '{self.name}'"
                )
            if config is not None:
                self._configs[name] = config

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

        if isinstance(cfg, dict):
            module_type = cfg.get("type")
            try:
                config_class = self._configs[module_type]
            except KeyError:
                raise KeyError(
                    f"No config registered for '{module_type}' in registry '{self.name}'. "
                    f"Available types: {self.list_configs()}"
                )

            cfg = config_class.model_validate(cfg)
        else:
            module_type = getattr(cfg, "type", None)

        if module_type is None:
            raise ValueError(f"Config object of type '{type(cfg).__name__}' has no 'type' attribute."
                             f" Expected a config with a 'type' field.")

        module_class = self.get(module_type)

        return module_class(
            cfg,
            *args,
            **kwargs
        )
    
    def list_modules(self):
        return list(self._modules.keys())

    def list_configs(self):
        return list(self._configs.keys())

    def __contains__(self, name):
        return name in self._modules
    
    def __repr__(self):
        return (
            f"Registry(name={self.name}), "
            f"items={list(self._modules.keys())}"
        )
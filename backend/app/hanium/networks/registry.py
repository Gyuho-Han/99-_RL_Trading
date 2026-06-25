"""네트워크 레지스트리 — 문자열 이름으로 네트워크 클래스를 찾아 생성한다.

사용법
------
# 등록 (각 네트워크 모듈 하단에서)
@NetworkRegistry.register("dnn")
class DNN(BaseNetwork): ...

# 생성 (train.py 등에서)
net = NetworkRegistry.create("dnn", input_dim=30, hidden_dims=[128, 64])
"""

from __future__ import annotations

from typing import Type

from .base_network import BaseNetwork


class NetworkRegistry:
    _registry: dict[str, Type[BaseNetwork]] = {}

    @classmethod
    def register(cls, name: str):
        """클래스 데코레이터 — 네트워크를 이름으로 등록한다."""
        def decorator(net_cls: Type[BaseNetwork]):
            if name in cls._registry:
                raise ValueError(f"Network '{name}' is already registered.")
            cls._registry[name] = net_cls
            return net_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseNetwork:
        """등록된 이름으로 네트워크 인스턴스를 생성한다."""
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys()) or "(없음)"
            raise KeyError(
                f"Network '{name}' not found. Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def list(cls) -> list[str]:
        """등록된 네트워크 이름 목록을 반환한다."""
        return list(cls._registry.keys())

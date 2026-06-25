"""에이전트 레지스트리 — 문자열 이름으로 에이전트 클래스를 찾아 생성한다.

사용법
------
# 등록 (각 에이전트 모듈 하단에서)
@AgentRegistry.register("dqn")
class DQNAgent(BaseAgent): ...

# 생성 (train.py 등에서)
agent = AgentRegistry.create("dqn", network=net, lr=0.001)
"""

from __future__ import annotations

from typing import Type

from .base_agent import BaseAgent


class AgentRegistry:
    _registry: dict[str, Type[BaseAgent]] = {}

    @classmethod
    def register(cls, name: str):
        """클래스 데코레이터 — 에이전트를 이름으로 등록한다."""
        def decorator(agent_cls: Type[BaseAgent]):
            if name in cls._registry:
                raise ValueError(f"Agent '{name}' is already registered.")
            cls._registry[name] = agent_cls
            return agent_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseAgent:
        """등록된 이름으로 에이전트 인스턴스를 생성한다."""
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys()) or "(없음)"
            raise KeyError(
                f"Agent '{name}' not found. Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def list(cls) -> list[str]:
        """등록된 에이전트 이름 목록을 반환한다."""
        return list(cls._registry.keys())

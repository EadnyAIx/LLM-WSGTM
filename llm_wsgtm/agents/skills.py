from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional


@dataclass
class Skill:
    name: str
    handler: Callable[..., Any]
    description: str = ""
    tags: tuple = ()


class SkillRegistry:
    def __init__(self):
        self._skills: Dict[str, Skill] = {}

    def register(self, name, handler, description="", tags: Iterable[str]=()):
        self._skills[name] = Skill(name=name, handler=handler, description=description, tags=tuple(tags))
        return handler

    def remove(self, name):
        self._skills.pop(name, None)

    def get(self, name) -> Optional[Skill]:
        return self._skills.get(name)

    def list(self):
        return list(self._skills.values())

    def invoke(self, name, **kwargs):
        skill = self.get(name)
        if skill is None:
            raise KeyError(name)
        return skill.handler(**kwargs)

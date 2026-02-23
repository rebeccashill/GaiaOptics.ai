# gaiaoptics/core/errors.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class ConfigError(Exception):
    path: str
    message: str
    hint: Optional[str] = None

    def __str__(self) -> str:
        out = [f"Config error at {self.path}:", f"  {self.message}"]
        if self.hint:
            out.append("hint:")
            out.append(f"  {self.hint}")
        return "\n".join(out)
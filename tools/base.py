from __future__ import annotations
import abc
from enum import Enum
from pathlib import Path
from typing import Any
from pydantic import BaseModel
from dataclasses import dataclass, field

@dataclass
class ToolInvocation:
    params: dict[str, Any]
    cwd: Path

@dataclass
class ToolResult:
    output: str
    success: bool
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

class ToolKind(str, Enum):
    READ = "read"
    WRITE = "write"
    SHELL = "shell"
    NETWORK = "network"
    MEMORY = "memory"
    MCP = "mcp"

class Tool(abc.ABC):
    name: str = "base_tool"
    description: str = "base tool"
    kind: ToolKind = ToolKind.READ
    
    def __init__(self) -> None:
        pass
      
    @property
    def schema(self) -> dict[str, Any] | type[BaseModel]:
      raise NotImplementedError("tool must define schema property or class attribute")
    
    @abc.abstractmethod
    async def execute(self, invocation: ToolInvocation) -> ToolResult:
      raise NotImplementedError("tool must implement execute method")
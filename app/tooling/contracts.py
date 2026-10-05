from dataclasses import dataclass,field
from typing import Any
from copy import deepcopy
import hashlib
import json


@dataclass
class CallContext:
    conversation_id:int|None=None
    user_id:str=''
    turn_id:str=''
    session_factory:Any=None
    raw_question:str=''
    budget:Any=None
    model:Any=None
    ticket_request:bool=False
    confirmation:Any=None
    trusted_results:dict=field(default_factory=dict)
    ticket_sources:tuple[str,...]|None=None


@dataclass(frozen=True)
class ToolDefinition:
    name:str
    description:str
    parameters:dict
    source:str
    mcp_server:str|None
    tool:Any

    @classmethod
    def from_tool(cls,tool,source='builtin',mcp_server=None):
        schema=tool.args_schema
        schema=schema if isinstance(schema,dict) else schema.model_json_schema()
        return cls(tool.name,tool.description,deepcopy(schema),source,mcp_server,tool)

    @property
    def key(self): return (self.source,self.mcp_server,self.name)

    @property
    def definition_hash(self):
        body=json.dumps([self.key,self.description,self.parameters],ensure_ascii=False,sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()


@dataclass(frozen=True)
class ToolPolicy:
    enabled:bool=True
    access:str='read'
    timeout_seconds:float=5
    max_retries:int=2
    retry_safe:bool=True


@dataclass(frozen=True)
class CatalogSnapshot:
    version:str
    definitions:tuple[ToolDefinition,...]
    def model_tools(self): return [d.tool for d in self.definitions]
    def get(self,name): return next((d for d in self.definitions if d.name==name),None)


@dataclass(frozen=True)
class ExecutionOutcome:
    kind:str
    payload:dict

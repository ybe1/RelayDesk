import hashlib
import importlib
import json
import logging
import re
from collections import Counter
from pathlib import Path
from .contracts import CatalogSnapshot,ToolDefinition
from .validation import check_schema

LOGGER=logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self,policy,plugin_path=None):
        self.policy=policy
        self.factories=[]
        self.plugin_path=Path(plugin_path) if plugin_path else None

    def register_builtin(self,factory):
        if factory not in self.factories: self.factories.append(factory)

    def _plugins(self):
        if not self.plugin_path: return []
        try:
            paths=json.loads(self.plugin_path.read_text(encoding='utf-8-sig'))
            if not isinstance(paths,list): raise ValueError('plugins must be array')
            factories=[]
            for path in paths:
                if not isinstance(path,str) or not re.fullmatch(r'app\.tools\.plugins\.[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*:[a-zA-Z_]\w*',path):
                    raise ValueError('untrusted plugin path')
                module,name=path.split(':')
                factory=getattr(importlib.import_module(module),name)
                if not callable(factory): raise ValueError('plugin factory not callable')
                factories.append(factory)
            return factories
        except Exception:
            LOGGER.exception('tool plugin registration failed')
            return []

    def snapshot(self,context,discovered=()):
        definitions=[]
        for factory in [*self.factories,*self._plugins()]:
            produced=factory(context)
            for tool in produced if isinstance(produced,list) else [produced]:
                definitions.append(ToolDefinition.from_tool(tool))
        definitions.extend(discovered)
        counts=Counter(d.name for d in definitions)
        accepted=[]
        for d in definitions:
            try:
                if counts[d.name]!=1: raise ValueError('duplicate tool name: '+d.name)
                if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',d.name): raise ValueError('invalid tool name')
                if not d.description.strip(): raise ValueError('missing tool description')
                if d.source not in {'builtin','mcp'} or (d.source=='mcp')!=bool(d.mcp_server): raise ValueError('invalid source')
                check_schema(d.parameters)
                if self.policy.lookup(d.key): accepted.append(d)
            except Exception:
                LOGGER.warning('tool registration rejected name=%s',d.name,exc_info=True)
        accepted.sort(key=lambda d:d.name)
        version=hashlib.sha256(''.join(d.definition_hash for d in accepted).encode()).hexdigest()
        return CatalogSnapshot(version,tuple(accepted))

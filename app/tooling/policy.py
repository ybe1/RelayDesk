import json
import logging
from pathlib import Path
from .contracts import ToolPolicy

LOGGER=logging.getLogger(__name__)


class PolicyStore:
    def __init__(self,path): self.path=Path(path)

    def lookup(self,key):
        try:
            rows=json.loads(self.path.read_text(encoding='utf-8-sig'))
            if not isinstance(rows,dict): raise ValueError('policy must be object')
            entry=rows.get(':'.join(v or '' for v in key))
            if entry is None: return None
            if not isinstance(entry,dict): raise ValueError('policy entry must be object')
            if set(entry)-{'enabled','access','timeout_seconds','max_retries','retry_safe'}: raise ValueError('unknown policy field')
            policy=ToolPolicy(**entry)
            if type(policy.enabled) is not bool or type(policy.retry_safe) is not bool: raise ValueError('boolean policy flag required')
            if policy.access not in {'read','write'}: raise ValueError('invalid access')
            if type(policy.max_retries) is not int or not 0<=policy.max_retries<=2: raise ValueError('invalid retries')
            if type(policy.timeout_seconds) not in (int,float) or not 0<policy.timeout_seconds<=120: raise ValueError('invalid timeout')
            if policy.access=='write':
                if key!=('builtin',None,'create_ticket'): return None
                policy=ToolPolicy(policy.enabled,'write',policy.timeout_seconds,0,False)
            if not policy.enabled: return None
            # A policy cannot downgrade the built-in write to read permission.
            if key==('builtin',None,'create_ticket') and policy.access!='write': return None
            return policy
        except (OSError,ValueError,TypeError):
            LOGGER.warning('tool policy invalid; denied',exc_info=True)
            return None

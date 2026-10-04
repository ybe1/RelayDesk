import json
from langchain_core.messages import ToolMessage

STATUS_LABELS={'delivered':'已签收','in_transit':'运输中','pending':'待处理','shipped':'已发货',
               'in_warranty':'在保','expired':'已过保','approved':'已通过','processing':'处理中'}


def error(code,message,**extra): return {'error':{'code':code,'message':message,**extra}}


def normalize_result(definition,raw):
    if isinstance(raw,ToolMessage):
        if raw.status=='error': return error('execution_failed','外部工具返回业务错误')
        raw=raw.content
    if isinstance(raw,str):
        try: raw=json.loads(raw)
        except ValueError: pass
    if isinstance(raw,list) and len(raw)==1 and isinstance(raw[0],dict) and raw[0].get('type')=='text':
        raw=raw[0].get('text','')
        try: raw=json.loads(raw)
        except (ValueError,TypeError): pass
    if raw is None or raw==[] or (isinstance(raw,dict) and raw.get('found') is False):
        return error('not_found','未查到匹配的业务记录')
    if isinstance(raw,dict) and raw.get('isError'):
        return error('execution_failed','外部工具返回错误')
    if isinstance(raw,dict) and 'error' in raw:
        detail=raw['error']
        return error('not_found' if isinstance(detail,dict) and detail.get('code')=='not_found' else 'execution_failed',
                     str(detail.get('message','工具返回错误') if isinstance(detail,dict) else detail)[:512])
    if isinstance(raw,dict):
        raw={k:v for k,v in raw.items() if not k.startswith('_') and k not in {'debug','internal_trace','access_token','api_key'}}
        if isinstance(raw.get('status'),str): raw['status']=STATUS_LABELS.get(raw['status'],raw['status'])
    return {'result':raw}


def serialize_result(payload):
    text=json.dumps(payload,ensure_ascii=False)
    if len(text.encode('utf-8'))>60000:
        return json.dumps(error('execution_failed','工具结果过大，无法完整提供，请缩小查询范围'),ensure_ascii=False)
    return text

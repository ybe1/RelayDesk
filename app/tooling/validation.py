from jsonschema import Draft202012Validator


def check_schema(schema):
    def walk(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in {'$ref','$dynamicRef'} and (not isinstance(item,str) or not item.startswith('#')):
                    raise ValueError('工具Schema只允许本地引用')
                walk(item)
        elif isinstance(value,list):
            for item in value: walk(item)
    walk(schema)
    Draft202012Validator.check_schema(schema)
    if schema.get('type')!='object': raise ValueError('工具参数必须是object')


def validate_arguments(schema,args):
    try:
        check_schema(schema)
        errors=sorted(Draft202012Validator(schema).iter_errors(args),key=lambda e:'/'.join(map(str,e.absolute_path)))
        return [{'path':'/'.join(map(str,e.absolute_path)), 'message':e.message} for e in errors]
    except Exception as exc:
        return [{'path':'','message':'参数Schema无法验证: '+str(exc)[:200]}]

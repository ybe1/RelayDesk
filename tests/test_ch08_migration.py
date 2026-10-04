import pytest


def test_schema_check_rejects_foreign_key_and_wrong_type():
    from scripts.migrate_ch08 import validate_shape,expected_shape
    correct=expected_shape()
    validate_shape(correct)
    wrong={**correct,'foreign_keys':[{'constrained_columns':['conversation_id']}]}
    with pytest.raises(RuntimeError): validate_shape(wrong)
    wrong={**correct,'columns':{**correct['columns'],'retry_count':('INTEGER',False,'0',None)}}
    with pytest.raises(RuntimeError): validate_shape(wrong)

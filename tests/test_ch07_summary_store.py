import pytest
from sqlalchemy import create_engine,text
from sqlalchemy.orm import sessionmaker

@pytest.fixture
def repo():
    from app.repositories.summaries import SummaryRepository
    engine=create_engine('sqlite://')
    with engine.begin() as c:
        c.execute(text('CREATE TABLE conversations (id INTEGER PRIMARY KEY,user_id TEXT,status TEXT,summary TEXT,summary_upto_msg_id INTEGER,layer1_from_msg_id INTEGER,created_at DATETIME,updated_at DATETIME)'))
        c.execute(text('CREATE TABLE conversation_summaries (id INTEGER PRIMARY KEY,conversation_id INTEGER,seq INTEGER,from_msg_id INTEGER,upto_msg_id INTEGER,content TEXT,created_at DATETIME DEFAULT CURRENT_TIMESTAMP,UNIQUE(conversation_id,seq))'))
        c.execute(text("INSERT INTO conversations(id,user_id,status) VALUES(1,'u','进行中')"))
    return SummaryRepository(sessionmaker(bind=engine))

def test_append_cas_and_immutable_segments(repo):
    assert repo.advance_layer1(1,0,4)
    assert not repo.advance_layer1(1,0,2)
    first=repo.append(1,0,4,'用户询问订单1001。')
    assert first.seq==1
    assert repo.append(1,0,4,'重复') is None
    assert repo.advance_layer1(1,4,8)
    second=repo.append(1,4,8,'用户询问退货条件。')
    anchors,segments=repo.load(1)
    assert anchors.summary_upto==8
    assert [(s.seq,s.content) for s in segments]==[(1,first.content),(2,second.content)]

def test_no_facts_advances_without_empty_segment(repo):
    repo.advance_layer1(1,0,2)
    assert repo.append(1,0,2,'') is None
    anchors,segments=repo.load(1)
    assert anchors.summary_upto==2 and not segments

def test_cannot_cover_unsummarized_recent_history(repo):
    assert repo.append(1,0,5,'订单1001') is None
    assert repo.load(1)[0].summary_upto==0

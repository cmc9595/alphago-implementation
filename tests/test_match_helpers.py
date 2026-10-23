import shlex
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from match import Engine, make_summary, wilson_interval


def test_gtp_transport_and_close():
    code="import sys\nfor line in sys.stdin:\n if line.strip()=='quit': break\n print('= ok\\n',flush=True)\n"
    e=Engine('gtp:'+shlex.quote(sys.executable)+' -u -c '+shlex.quote(code),7.5,timeout=5)
    assert e.cmd('name')=='ok'
    e.close()
    assert e.proc.poll() is not None


def test_gtp_timeout_terminates_child():
    code='import time; time.sleep(30)'
    with pytest.raises(TimeoutError):
        Engine('gtp:'+shlex.quote(sys.executable)+' -u -c '+shlex.quote(code),7.5,timeout=.3)


def test_capped_games_are_not_counted_as_wins():
    r={'game':0,'pair':0,'a_won':True,'a_score':1.,'moves':500,'capped':True,
       'move_seconds':{'A':[.4],'B':[.3]},'search_visits':{'A':[10],'B':[20]}}
    s=make_summary([r],'a','b',1)
    assert s['completed_games']==1 and s['games']==0 and s['capped_games']==1
    assert s['A_win_rate'] is None
    assert wilson_interval(4,4)==[.51,1.]

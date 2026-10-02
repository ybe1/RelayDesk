import json
import os
import subprocess
from pathlib import Path

def test_sidebar_in_headless_browser_with_mocked_http():
    env={**os.environ,'PYTHONIOENCODING':'utf-8'}
    result=subprocess.run(['uv','run','--with','playwright','python','-m','scripts.verify_ch07_browser'],
        cwd=Path(__file__).resolve().parents[1],env=env,capture_output=True,text=True,encoding='utf-8',timeout=60)
    assert result.returncode==0,result.stderr+result.stdout
    report=json.loads(result.stdout.strip().splitlines()[-1])
    assert report['status']=='passed'
    assert {'full_history','stream_lock','feedback_restore','sidebar_failure'}<=set(report['checks'])

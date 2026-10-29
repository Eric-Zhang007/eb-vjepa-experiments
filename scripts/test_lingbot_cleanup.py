import hashlib,json,subprocess,tempfile
from pathlib import Path
script=Path(__file__).with_name('lingbot_delete_verified.ps1').resolve()
with tempfile.TemporaryDirectory(prefix='lingbot_cleanup_test_') as tmp:
    root=Path(tmp); data=root/'lingbot-fixture'; data.mkdir(); task=root/'receipts';task.mkdir()
    a=data/'original.bin';b=data/'duplicate.bin';a.write_bytes(b'test content\x00'*1024);b.write_bytes(a.read_bytes())
    def row(p):
        s=p.stat();return dict(path=str(p),size=s.st_size,mtime_ns=s.st_mtime_ns)
    manifest=task/'verified_duplicates.jsonl';status=task/'status.json'
    manifest.write_text(json.dumps(dict(keep=row(a),delete=row(b),sha256=hashlib.sha256(a.read_bytes()).hexdigest()))+'\n')
    status.write_text(json.dumps(dict(phase='complete')))
    cmd=['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(script),'-TaskDir',str(task),'-RootDirectory',str(root)]
    result=subprocess.run(cmd,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert a.exists() and not b.exists()
    assert json.loads((task/'deletions.jsonl').read_text(encoding='utf-8-sig'))['size']==a.stat().st_size
    # A modified candidate must be refused; the original must remain intact.
    c=data/'changed.bin';c.write_bytes(b'first');new_task=root/'negative';new_task.mkdir()
    (new_task/'verified_duplicates.jsonl').write_text(json.dumps(dict(keep=row(a),delete=row(c),sha256=hashlib.sha256(a.read_bytes()).hexdigest()))+'\n')
    (new_task/'status.json').write_text(json.dumps(dict(phase='complete')))
    c.write_bytes(b'changed size')
    cmd[cmd.index('-TaskDir')+1]=str(new_task)
    result=subprocess.run(cmd,capture_output=True,text=True)
    assert result.returncode!=0 and c.exists() and a.exists(),result.stdout
print('PASS: exact duplicate removed, keeper retained, changed candidate rejected')

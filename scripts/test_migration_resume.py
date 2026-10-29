import hashlib,json,paramiko
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('112.45.47.40',username='jiachangzhang',key_filename='C:/Users/zjc/.ssh/dnft4rp_ed25519',banner_timeout=90)
cmd='python3 /data/users/jiachangzhang/eb_vjepa/scripts/migration_cached_stream_remote.py /data/users/jiachangzhang/eb_vjepa/artifacts/cluster_migration_20261026/stream_test.json '
ch=c.get_transport().open_session();ch.exec_command(cmd+'0');head=ch.recv(65536);ch.close()
ch=c.get_transport().open_session();ch.exec_command(cmd+str(len(head)));h=hashlib.sha256(head);n=len(head)
while True:
    b=ch.recv(1024**2)
    if not b:break
    h.update(b);n+=len(b)
err=b''
while True:
    b=ch.recv_stderr(65536)
    if not b:break
    err+=b
assert ch.recv_exit_status()==0,err
r=json.loads(err);assert n==r['bytes'] and h.hexdigest()==r['sha256']
print('PASS: intentionally interrupted stream resumed at byte',len(head),'and full checksum matched;',n,'bytes')
c.close()

"""Shared, lossless CPU RAM cache for frozen encoder outputs (Unix socket)."""
import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import pickle
import socket
import socketserver
import struct
import threading
import time
import zstandard as zstd


def receive(sock):
    def exact(n):
        parts = bytearray()
        while len(parts) < n:
            chunk = sock.recv(min(n - len(parts), 8 * 1024 * 1024))
            if not chunk:
                raise EOFError()
            parts.extend(chunk)
        return parts
    size = struct.unpack('!Q', exact(8))[0]
    if size > 256 * 1024 * 1024:
        raise ValueError('Oversized cache message')
    return pickle.loads(exact(size))


def send(sock, value):
    payload = pickle.dumps(value, protocol=5)
    sock.sendall(struct.pack('!Q', len(payload)))
    sock.sendall(payload)


class Client:
    def __init__(self, path):
        self.path, self.sock = path, None

    def request(self, op, value=None):
        if self.sock is None:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(60)
            self.sock.connect(self.path)
        send(self.sock, (op, value))
        reply = receive(self.sock)
        if isinstance(reply, dict) and 'error' in reply:
            raise RuntimeError(reply['error'])
        return reply

    def get(self, keys):
        values = self.request('get', keys)
        decoder = zstd.ZstdDecompressor()
        return [None if v is None else decoder.decompress(v) for v in values]

    def put(self, pairs):
        encoder = zstd.ZstdCompressor(level=1)
        return self.request('put', [(key, encoder.compress(data)) for key, data in pairs])


class Store:
    def __init__(self, capacity):
        self.capacity, self.used = capacity, 0
        self.values = OrderedDict()
        self.lock = threading.Lock()
        self.hits = self.misses = self.evictions = 0

    def stats(self):
        return dict(items=len(self.values), bytes=self.used, capacity=self.capacity,
                    hits=self.hits, misses=self.misses, evictions=self.evictions)

    def request(self, op, payload):
        with self.lock:
            if op == 'get':
                result = []
                for key in payload:
                    data = self.values.get(key)
                    if data is None:
                        self.misses += 1
                    else:
                        self.hits += 1
                        self.values.move_to_end(key)
                    result.append(data)
                return result
            if op == 'put':
                for key, data in payload:
                    if key in self.values:
                        continue
                    # Leave at least 64 GiB available for training and checkpoint staging.
                    with open('/proc/meminfo') as stream:
                        available = next(int(l.split()[1])*1024 for l in stream if l.startswith('MemAvailable:'))
                    while self.values and (self.used + len(data) > self.capacity or available < 64*1024**3):
                        _, old = self.values.popitem(last=False)
                        self.used -= len(old)
                        available += len(old)
                        self.evictions += 1
                    if len(data) <= self.capacity and available >= 64*1024**3:
                        self.values[key] = data
                        self.used += len(data)
                return self.stats()
            if op == 'stats':
                return self.stats()
            raise ValueError(op)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--socket', required=True)
    parser.add_argument('--capacity-gib', type=float, default=160)
    parser.add_argument('--status', required=True)
    args = parser.parse_args()
    store = Store(int(args.capacity_gib * 1024**3))
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                while True:
                    op, payload = receive(self.request)
                    send(self.request, store.request(op, payload))
            except (EOFError, ConnectionError):
                return
    class Server(socketserver.ThreadingUnixStreamServer):
        daemon_threads = True
    path = Path(args.socket)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never unlink a live service's socket.
    if path.exists():
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(str(path))
        except ConnectionRefusedError:
            path.unlink()
        else:
            raise RuntimeError('Cache service already running')
        finally:
            probe.close()
    server = Server(str(path), Handler)
    os.chmod(path, 0o600)
    def report():
        while True:
            stats = store.request('stats', None)
            stats.update(pid=os.getpid(), time=time.time())
            target = Path(args.status)
            tmp = target.with_suffix('.tmp')
            tmp.write_text(json.dumps(stats))
            tmp.replace(target)
            time.sleep(10)
    threading.Thread(target=report, daemon=True).start()
    print(json.dumps(dict(status='ready', socket=str(path), capacity=store.capacity)), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()

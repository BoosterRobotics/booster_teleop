"""Read-only robot-local health. No SDK, interface scans, or management peers.

Network is business-stream freshness, NOT bandwidth or ICMP quality. PICO's
63901 accepted TCP connection is owned by RoboticsService (verified onboard).
PN-Link's receiver does not expose the peer: use the configured suit IP.
"""
from __future__ import annotations

import math
import re
import copy
import ipaddress
import os
from pathlib import Path
import socket
import subprocess
import threading
import time


def device_ip(value):
    """Only explicit numeric unicast device addresses; never DNS/routes/SSH."""
    try:
        address = ipaddress.ip_address(value)
        if address.is_loopback or address.is_unspecified or address.is_multicast or address.is_link_local:
            return None
        return str(address)
    except (ValueError, TypeError):
        return None


def pico_peer(proc=Path('/proc')):
    """Unique ESTABLISHED :63901 peer, inode owned by RoboticsService only.

    /proc socket tables alone are insufficient: unrelated processes could use
    the same port. Permission failures and multiple headsets are unknown.
    """
    inodes = set()
    try:
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if entry.joinpath('comm').read_text().strip() not in {'RoboticsService', 'RoboticsServiceProcess'}:
                    continue
                for fd in entry.joinpath('fd').iterdir():
                    try:
                        link = os.readlink(fd)
                        if link.startswith('socket:['):
                            inodes.add(link[8:-1])
                    except OSError:
                        pass
            except OSError:
                continue
    except OSError:
        return None
    peers = set()
    for table, family in [('tcp', socket.AF_INET), ('tcp6', socket.AF_INET6)]:
        try:
            lines = (proc/'net'/table).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10 or parts[3] != '01' or parts[9] not in inodes:
                continue
            if parts[1].split(':')[1].upper() != 'F99D':
                continue
            try:
                raw = bytes.fromhex(parts[2].split(':')[0])
                raw = b''.join(raw[i:i+4][::-1] for i in range(0, len(raw), 4))
                address = ipaddress.ip_address(socket.inet_ntop(family, raw))
                address = getattr(address, 'ipv4_mapped', None) or address
                host = device_ip(str(address))
                if host:
                    peers.add(host)
            except (ValueError, OSError):
                continue
    return next(iter(peers)) if len(peers) == 1 else None


def wireless(proc=Path('/proc')):
    """Local-interface auxiliary RSSI only, not motion-device signal strength."""
    result = []
    try:
        for line in (proc/'net/wireless').read_text().splitlines()[2:]:
            name, _, rest = line.partition(':')
            fields = rest.split()
            if len(fields) >= 3:
                level = float(fields[2].rstrip('.'))
                if -150 < level < 0:
                    result.append({'interface': name.strip(), 'rssi_dbm': level})
    except (OSError, ValueError):
        pass
    return result


class SystemHealth:
    """Single 1 Hz worker/cache shared by HTTP clients; no work in snapshot()."""
    def __init__(self, device_host=None, proc=Path('/proc')):
        if device_host and not device_ip(device_host):
            raise ValueError('--device-host requires a numeric unicast motion device IP')
        self.device_host = device_host
        self.proc = proc
        self.cpu = CpuSampler()
        self._lock = threading.Lock()
        self._process_lock = threading.Lock()
        self._context = {}
        self._identity = None
        self._generation = 0
        self._data = {}
        self._thread = None
        self._process = None
        self._stop = threading.Event()

    def update(self, snapshot):
        session = snapshot.get('session') or {}
        identity = tuple(session.get(k) for k in ('name', 'alive', 'source', 'replay'))
        with self._lock:
            self._context = {'session': dict(session), 'panes': copy.deepcopy(snapshot.get('panes', [])), 'updated': time.monotonic()}
            if identity != self._identity:
                self._identity = identity
                self._generation += 1
                # Never expose a former device's ping during a switch.
                self._data = {k:v for k,v in self._data.items() if k in ('cpu', 'wireless', 'sampled_mono')}

    def snapshot(self):
        now = time.monotonic()
        with self._lock:
            result = copy.deepcopy(self._data)
            context = self._context
            session = context.get('session', {})
            age = now-result.get('sampled_mono', now) if 'sampled_mono' in result else None
            context_age = now-context.get('updated', now)
        result['age_s'] = age
        result['context_age_s'] = context_age
        result.setdefault('cpu', {'total':None, 'cores':{}})
        result.setdefault('wireless', [])
        result.setdefault('target', None)
        offline = session.get('replay') == 'offline'
        result['offline'] = offline
        if result.get('network') == 'online' and now-result.get('stream_observed_mono', 0) >= 2.5:
            result['network'] = 'weak'
        if offline:
            result.update(target=None, network='na', ping={'state':'na', 'ms':None, 'color':'gray'})
        elif age is None or age > 2.5 or context_age > 3 or 'ping' not in result:
            result.update(target=None, network='unknown', ping={'state':'unknown', 'ms':None, 'color':'gray'})
        if age is None or age > 2.5:
            result['cpu'] = {'total':None, 'cores':{}}
            result['wireless'] = []
        return result

    def _ping(self, host):
        unknown = {'state':'unknown', 'ms':None, 'color':'gray'}
        try:
            with self._process_lock:
                if self._stop.is_set():
                    return unknown
                self._process = subprocess.Popen(['ping', '-n', '-c', '1', '-W', '1', host], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env={**os.environ, 'LC_ALL':'C'})
                process = self._process
            try:
                output, _ = process.communicate(timeout=.9)
                return parse_ping(process.returncode, output)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                return {'state':'unreachable', 'ms':None, 'color':'red'}
            finally:
                with self._process_lock:
                    self._process = None
        except OSError:
            return unknown

    def sample_once(self):
        begin = time.monotonic()
        with self._lock:
            context = copy.deepcopy(self._context)
            generation = self._generation
        try:
            cpu = self.cpu.sample((self.proc/'stat').read_text())
        except OSError:
            cpu = self.cpu.sample('')
        result = {'cpu':cpu, 'wireless':wireless(self.proc), 'target':None, 'network':'unknown', 'ping':{'state':'unknown', 'ms':None, 'color':'gray'}}
        session = context.get('session', {})
        online = session.get('alive') and session.get('replay') == 'online' and time.monotonic()-context.get('updated', 0) < 3
        if online:
            source = session.get('source')
            host = self.device_host
            if not host and source == 'mocap':
                host = device_ip(os.getenv('BOOSTER_MOCAP_SERVER_IP'))
            elif not host and source == 'pico':
                host = pico_peer(self.proc)
            result['target'] = host
            publisher = next((p for p in context.get('panes', []) if p.get('role') == 'publisher'), {})
            age = publisher.get('age_s')
            # Existing source body/sent telemetry, NOT downstream consumption.
            hz = publisher.get('hz') or {}
            rate = hz.get('body', hz.get('sent'))
            if host and age is not None:
                result['stream_observed_mono'] = context.get('updated', begin)-age
                fresh = age + time.monotonic()-context.get('updated', begin) < 2.5 and isinstance(rate, (int,float)) and rate > 0 and not publisher.get('dead') and not publisher.get('service_down')
                result['network'] = 'online' if fresh else 'weak'
            if host and not self._stop.is_set():
                result['ping'] = self._ping(host)
        result['sampled_mono'] = begin  # Includes ping time: never rejuvenate old samples.
        result['sample_duration_ms'] = (time.monotonic()-begin)*1000
        with self._lock:
            if generation == self._generation:
                self._data = result

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name='dashboard-system-health', daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            begin = time.monotonic()
            self.sample_once()
            self._stop.wait(max(0, 1-(time.monotonic()-begin)))

    def stop(self):
        self._stop.set()
        with self._process_lock:
            if self._process is not None and self._process.poll() is None:
                self._process.kill()
        if self._thread is not None:
            self._thread.join(timeout=2)



class CpuSampler:
    def __init__(self):
        self.previous = {}

    def sample(self, text):
        current, values = {}, {}
        for line in text.splitlines():
            parts = line.split()
            if not parts or not re.fullmatch(r'cpu\d*', parts[0]):
                continue
            try:
                ticks = tuple(int(x) for x in parts[1:9])  # guest already in user/nice
                if len(ticks) < 4 or any(x < 0 for x in ticks):
                    continue
            except ValueError:
                continue
            name = parts[0]
            current[name] = ticks
            old = self.previous.get(name)
            value = None
            if old is not None and len(old) == len(ticks):
                delta = [a-b for a, b in zip(ticks, old)]
                total = sum(delta)
                if min(delta) >= 0 and total > 0:
                    idle = delta[3] + (delta[4] if len(delta) > 4 else 0)
                    value = round(100 * (total-idle)/total, 1)
            values[name] = value
        self.previous = current
        return {'total': values.get('cpu'), 'cores': {k:v for k,v in values.items() if k != 'cpu'}}


def ping_color(state, ms):
    if state == 'unreachable':
        return 'red'
    if state != 'ok' or ms is None or not math.isfinite(ms) or ms < 0:
        return 'gray'
    return 'blue' if ms < 5 else 'green' if ms < 10 else 'orange' if ms <= 20 else 'red'


def parse_ping(returncode, output):
    if re.search(r'not permitted|permission denied|operation not supported', output, re.I):
        return {'state':'unknown', 'ms':None, 'color':'gray'}
    match = re.search(r'time[=<]\s*([0-9.]+)\s*ms', output)
    ms = None
    if returncode == 0 and match:
        try:
            ms = float(match[1])
        except ValueError:
            pass
    unreachable = returncode == 1 or (returncode > 0 and re.search(r'unreachable|no route to host|packet loss', output, re.I))
    state = 'ok' if ms is not None and math.isfinite(ms) else ('unreachable' if unreachable else 'unknown')
    return {'state': state, 'ms': ms if state == 'ok' else None, 'color': ping_color(state, ms)}

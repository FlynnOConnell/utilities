"""
Network health checks for the remote data paths.

Data lives two network hops away from a workstation: the SSH link out to the
lab server, and the CIFS mount that server pulls from (biohpc/lamella). A slow
`mbo view` on a remote file can be either hop, so the checks here measure them
separately, and report the local NIC line rate alongside, since a throughput
number only means something next to the link it had to fit through.

Nothing here writes to the remote host: throughput is measured against
/dev/zero and /dev/null, and reads come from a file the caller names.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any

# a probe is worth the wall-clock it costs; these are sized so a full run is
# well under a minute on a link that is behaving, and bails early on one that
# is not.
_SSH_CONNECT_TIMEOUT = 15
_PROBE_TIMEOUT = 180
_DEFAULT_SIZE_MB = 512
_LATENCY_SAMPLES = 12
_STAT_SAMPLES = 20

# sshd's default MaxStartups (10:30:100) begins randomly refusing connections
# once 10 are unauthenticated in flight, so a probe run that opens several in
# quick succession loses some to the dice rather than to the network. Retry
# rather than report a healthy link as broken.
_SSH_RETRIES = 3
_SSH_RETRY_BACKOFF = 0.75

# below this fraction of the local NIC's line rate, the link is worth a look
_LINK_RATE_WARN_FRAC = 0.35
# a campus link should be well under this; above it, interactive work drags
_HIGH_RTT_MS = 20
# at or above this fraction of line rate, the NIC is the ceiling
_LINK_SATURATED_FRAC = 0.6
# parallel has to beat a single stream by this much to count as real gain
_PARALLEL_GAIN_FRAC = 1.25
# storage must outrun the link by this much before the link is the culprit
_STORAGE_FASTER_FRAC = 1.5
# per-op latencies above these make file browsing feel broken
_SLOW_LISTDIR_MS = 250
_SLOW_STAT_MS = 20


def _no_window_flags() -> int:
    """CREATE_NO_WINDOW on Windows so subprocess calls don't flash a console."""
    if sys.platform == "win32":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0


def _ssh_base(host: str) -> list[str]:
    """Ssh argv that never prompts, so a bad key fails fast instead of hanging."""
    return [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={_SSH_CONNECT_TIMEOUT}",
        "-o", "StrictHostKeyChecking=accept-new",
        host,
    ]


def _shq(text: str) -> str:
    """Single-quote a path for the remote shell."""
    return "'" + str(text).replace("'", "'\\''") + "'"


def _run_capture(cmd: list[str], timeout: int = 30) -> str | None:
    """Run a command, return stdout, or None on any failure."""
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=_no_window_flags(),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _timed(cmd: list[str], timeout: int = _PROBE_TIMEOUT) -> tuple[int | None, float]:
    """Run a command discarding stdout, return (returncode, elapsed_seconds)."""
    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            creationflags=_no_window_flags(),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None, time.perf_counter() - start
    return result.returncode, time.perf_counter() - start


def _timed_retry(cmd: list[str],
                 timeout: int = _PROBE_TIMEOUT) -> tuple[int | None, float]:
    """_timed, retried past sshd's startup throttle.

    Only the successful attempt's elapsed time is returned, so a refused
    connection costs wall-clock but never contaminates the measurement.
    """
    for attempt in range(_SSH_RETRIES):
        rc, elapsed = _timed(cmd, timeout=timeout)
        if rc == 0:
            return rc, elapsed
        if attempt < _SSH_RETRIES - 1:
            time.sleep(_SSH_RETRY_BACKOFF * (attempt + 1))
    return None, 0.0


def _timed_upload(cmd: list[str], size_mb: int,
                  timeout: int = _PROBE_TIMEOUT) -> tuple[int | None, float]:
    """Feed size_mb of zeros to a command's stdin, return (returncode, elapsed)."""
    start = time.perf_counter()
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_no_window_flags(),
        )
    except (FileNotFoundError, OSError):
        return None, time.perf_counter() - start

    chunk = b"\0" * (1 << 20)
    try:
        for _ in range(size_mb):
            proc.stdin.write(chunk)
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass

    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return None, time.perf_counter() - start
    return rc, time.perf_counter() - start


def _ssh_capture(host: str, remote_cmd: str,
                 timeout: int = _PROBE_TIMEOUT) -> str | None:
    """Run a remote command and return its stdout, or None on any failure.

    Retried, because a refused connection here is usually sshd's startup
    throttle rather than a real fault (see _SSH_RETRIES).
    """
    cmd = _ssh_base(host) + [remote_cmd]
    for attempt in range(_SSH_RETRIES):
        out = _run_capture(cmd, timeout=timeout)
        if out:
            return out.strip()
        if attempt < _SSH_RETRIES - 1:
            time.sleep(_SSH_RETRY_BACKOFF * (attempt + 1))
    return None


def ssh_available() -> bool:
    return shutil.which("ssh") is not None


def resolve_hostname(host: str) -> str | None:
    """The HostName ssh_config maps this alias to, so ping hits the real host."""
    out = _run_capture(["ssh", "-G", host], timeout=15)
    if not out:
        return None
    for line in out.splitlines():
        if line.lower().startswith("hostname "):
            return line.split(None, 1)[1].strip()
    return None


# --------------------------------------------------------------------------
# local link context
# --------------------------------------------------------------------------

def _parse_link_speed(text: str) -> float | None:
    """'2.5 Gbps' -> 2500.0 (Mbit/s). Windows also reports a bare bits/s int."""
    text = text.strip()
    if not text:
        return None
    parts = text.split()
    try:
        value = float(parts[0])
    except (ValueError, IndexError):
        return None
    unit = parts[1].lower() if len(parts) > 1 else ""
    if unit.startswith("g"):
        return value * 1000
    if unit.startswith("m"):
        return value
    if unit.startswith("k"):
        return value / 1000
    return value / 1e6  # bare number: bits per second


def local_link_mbps() -> dict[str, Any] | None:
    """Line rate of the local NIC carrying the route, in Mbit/s.

    Throughput without this is uninterpretable: 230 MiB/s is a saturated
    2.5GbE port and a badly underperforming 10GbE one.
    """
    if sys.platform == "win32":
        # skip tunnels: Tailscale reports a fictional 100 Gbps link speed
        script = (
            "Get-NetAdapter | Where-Object {$_.Status -eq 'Up' -and "
            "$_.InterfaceDescription -notmatch 'Tailscale|Loopback|Virtual|VPN|WAN'} | "
            "Select-Object -First 1 Name,LinkSpeed,InterfaceDescription | "
            "ConvertTo-Json"
        )
        out = _run_capture(
            ["powershell.exe", "-NoProfile", "-Command", script], timeout=30
        )
        if not out:
            return None
        try:
            data = json.loads(out)
        except (json.JSONDecodeError, ValueError):
            return None
        if isinstance(data, list):
            data = data[0] if data else None
        if not data:
            return None
        speed = str(data.get("LinkSpeed", ""))
        return {
            "name": data.get("Name"),
            "description": data.get("InterfaceDescription"),
            "link_speed": speed,
            "mbps": _parse_link_speed(speed),
        }

    for path in sorted(glob.glob("/sys/class/net/*/speed")):
        iface = os.path.basename(os.path.dirname(path))
        if iface.startswith(("lo", "tailscale", "docker", "veth", "virbr")):
            continue
        try:
            with open(path) as fh:
                mbps = float(fh.read().strip())
        except (OSError, ValueError):
            continue
        if mbps <= 0:
            continue
        return {
            "name": iface,
            "description": iface,
            "link_speed": f"{mbps / 1000:g} Gbps" if mbps >= 1000 else f"{mbps:g} Mbps",
            "mbps": mbps,
        }
    return None


# --------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------

def _ping_rtts(target: str, samples: int) -> list[float]:
    """Parse per-reply RTTs out of the platform ping."""
    if sys.platform == "win32":
        cmd = ["ping", "-n", str(samples), target]
    else:
        cmd = ["ping", "-c", str(samples), "-i", "0.2", target]
    out = _run_capture(cmd, timeout=samples * 2 + 20)
    if not out:
        return []

    rtts: list[float] = []
    for line in out.splitlines():
        low = line.lower()
        marker = low.find("time")
        if marker == -1:
            continue
        # the RTT is the number right after the 'time' marker: 'time=0.4 ms',
        # 'time<1ms', 'time=1ms'
        tail = low[marker + 4:].lstrip("=<").strip()
        number = ""
        for ch in tail:
            if ch.isdigit() or ch == ".":
                number += ch
            else:
                break
        if not number:
            continue
        try:
            rtts.append(float(number))
        except ValueError:
            continue
    return rtts


def _ssh_rtts(host: str, samples: int) -> list[float]:
    """Time `true` over SSH. Includes handshake, so it overstates the wire."""
    rtts: list[float] = []
    for _ in range(samples):
        rc, elapsed = _timed(
            _ssh_base(host) + ["true"], timeout=_SSH_CONNECT_TIMEOUT + 5
        )
        if rc == 0:
            rtts.append(elapsed * 1000)
    return rtts


def _ssh_port(host: str) -> int:
    """The Port ssh_config maps this alias to; 22 if it says nothing."""
    out = _run_capture(["ssh", "-G", host], timeout=15)
    if out:
        for line in out.splitlines():
            if line.lower().startswith("port "):
                try:
                    return int(line.split(None, 1)[1].strip())
                except (ValueError, IndexError):
                    break
    return 22


def _tcp_rtts(target: str, port: int, samples: int) -> list[float]:
    """TCP handshake RTT, in ms.

    Windows ping only reports whole milliseconds, which on a campus link
    rounds every sample to the same integer and reports zero jitter. A
    connect() is timed at the clock's real resolution and traverses the same
    path the data will, so it measures what actually matters here.
    """
    import socket

    rtts: list[float] = []
    for _ in range(samples):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(_SSH_CONNECT_TIMEOUT)
        start = time.perf_counter()
        try:
            sock.connect((target, port))
            rtts.append((time.perf_counter() - start) * 1000)
        except (TimeoutError, OSError):
            pass
        finally:
            sock.close()
        # sshd counts these as unauthenticated connections; pace them so the
        # latency probe does not trip the throttle the retries exist for
        time.sleep(0.05)
    return rtts


def probe_latency(host: str, samples: int = _LATENCY_SAMPLES) -> dict[str, Any]:
    """Round-trip latency to the SSH host, over ICMP where it is permitted.

    ICMP is the closest thing to a pure network RTT available here. Windows
    only reports it to whole milliseconds, so on a fast campus link expect a
    coarse figure and a jitter near zero; :func:`probe_tcp_connect` carries
    the fine-grained number. Falls back to a TCP handshake, then to a trivial
    SSH command, which is an upper bound rather than a substitute.
    """
    target = resolve_hostname(host) or host
    rtts = _ping_rtts(target, samples)
    method = "icmp"
    if not rtts:
        # campus firewalls drop ICMP more often than they drop SSH
        rtts = _tcp_rtts(target, _ssh_port(host), min(samples, 6))
        method = "tcp-handshake"
    if not rtts:
        rtts = _ssh_rtts(host, min(samples, 5))
        method = "ssh-exec"
    if not rtts:
        return {"ok": False, "method": None, "target": target}
    return {
        "ok": True,
        "method": method,
        "target": target,
        "samples": len(rtts),
        "min_ms": min(rtts),
        "avg_ms": statistics.fmean(rtts),
        "max_ms": max(rtts),
        # jitter matters more than average for interactive file browsing
        "jitter_ms": statistics.stdev(rtts) if len(rtts) > 1 else 0.0,
    }


def probe_tcp_connect(host: str, samples: int = 6) -> dict[str, Any]:
    """Time to open a TCP connection to the SSH port.

    Distinct from network RTT: it includes sshd accepting the connection, so
    it is the per-connection cost paid by every new scp/rsync/sshfs session,
    and it is what makes many small transfers slower than one large one.
    """
    target = resolve_hostname(host) or host
    rtts = _tcp_rtts(target, _ssh_port(host), samples)
    if not rtts:
        return {"ok": False, "target": target}
    return {
        "ok": True,
        "target": target,
        "samples": len(rtts),
        "min_ms": min(rtts),
        "avg_ms": statistics.fmean(rtts),
        "max_ms": max(rtts),
    }


def probe_ssh_throughput(host: str, size_mb: int = _DEFAULT_SIZE_MB) -> dict[str, Any]:
    """Throughput both directions over SSH, against /dev/zero and /dev/null.

    This is the number that governs scp/rsync/sshfs, and it folds in cipher
    cost, which is the point: that cost is paid on every real transfer.
    """
    result: dict[str, Any] = {"size_mb": size_mb}

    rc, elapsed = _timed_retry(
        _ssh_base(host) + [f"dd if=/dev/zero bs=1M count={size_mb} status=none"]
    )
    result["download_mibs"] = (size_mb / elapsed) if rc == 0 and elapsed > 0 else None

    rc, elapsed = _timed_upload(_ssh_base(host) + ["cat > /dev/null"], size_mb)
    result["upload_mibs"] = (size_mb / elapsed) if rc == 0 and elapsed > 0 else None
    return result


def probe_parallel_streams(host: str, streams: int = 4,
                           size_mb: int = 256) -> dict[str, Any]:
    """Aggregate throughput across N concurrent SSH streams.

    The diagnostic that separates a saturated link from a slow protocol: if
    the aggregate beats a single stream, the link had headroom and something
    per-stream was the limit; if it does not, the link itself is the ceiling.
    """
    procs = []
    start = time.perf_counter()
    for _ in range(streams):
        try:
            procs.append(subprocess.Popen(
                _ssh_base(host) + [
                    f"dd if=/dev/zero bs=1M count={size_mb} status=none"
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=_no_window_flags(),
            ))
        except (FileNotFoundError, OSError):
            break
    if not procs:
        return {"ok": False}

    ok = True
    for proc in procs:
        try:
            if proc.wait(timeout=_PROBE_TIMEOUT) != 0:
                ok = False
        except subprocess.TimeoutExpired:
            proc.kill()
            ok = False
    elapsed = time.perf_counter() - start
    if not ok or elapsed <= 0:
        return {"ok": False}

    total = size_mb * len(procs)
    return {
        "ok": True,
        "streams": len(procs),
        "size_mb": total,
        "aggregate_mibs": total / elapsed,
    }


def probe_remote_mount(host: str, path: str) -> dict[str, Any]:
    """What the remote path actually is: filesystem type, server, mount options.

    A CIFS mount with a low actimeo explains slow directory listings far
    better than any bandwidth number will.
    """
    quoted = _shq(path)
    out = _ssh_capture(
        host,
        f"df -PT {quoted} 2>/dev/null | tail -1; echo '---'; "
        f"findmnt -no SOURCE,FSTYPE,OPTIONS --target {quoted} 2>/dev/null",
        timeout=60,
    )
    if not out:
        return {"ok": False, "path": path}

    df_part, _, mnt_part = out.partition("---")
    info: dict[str, Any] = {"ok": True, "path": path}
    fields = df_part.split()
    if len(fields) >= 7:
        info["source"] = fields[0]
        info["fstype"] = fields[1]
        info["size"] = fields[2]
        info["used_pct"] = fields[5]
        info["mountpoint"] = fields[6]
    mnt = mnt_part.strip().split(None, 2)
    if len(mnt) >= 2:
        info.setdefault("source", mnt[0])
        info["fstype"] = mnt[1]
        if len(mnt) > 2:
            info["options"] = mnt[2]
    return info


def probe_remote_read(host: str, path: str, size_mb: int = _DEFAULT_SIZE_MB,
                      skip_mb: int = 0) -> dict[str, Any]:
    """Read speed of the remote storage as seen by the remote host itself.

    ``skip_mb`` reads from an offset to sidestep page cache from a prior run;
    it is the difference between measuring the storage and measuring RAM.
    """
    quoted = _shq(path)
    out = _ssh_capture(
        host,
        f"test -r {quoted} || {{ echo UNREADABLE; exit 0; }}; "
        f"s=$(date +%s%N); "
        f"dd if={quoted} of=/dev/null bs=1M skip={skip_mb} count={size_mb} "
        f"status=none 2>/dev/null; "
        f"e=$(date +%s%N); echo $(( (e-s)/1000000 ))",
    )
    if not out:
        # the probe never ran; saying "unreadable" would blame the file for
        # what is actually a connection failure
        return {"ok": False, "path": path, "reason": "no response over ssh"}
    if "UNREADABLE" in out:
        return {"ok": False, "path": path, "reason": "not readable by remote user"}
    try:
        ms = int(out.splitlines()[-1].strip())
    except (ValueError, IndexError):
        return {"ok": False, "path": path, "reason": "unparseable"}
    if ms <= 0:
        return {"ok": False, "path": path, "reason": "too-fast-to-time"}
    return {
        "ok": True,
        "path": path,
        "size_mb": size_mb,
        "skip_mb": skip_mb,
        "read_mibs": size_mb * 1000 / ms,
    }


def probe_end_to_end(host: str, path: str, size_mb: int = _DEFAULT_SIZE_MB,
                     skip_mb: int = 0) -> dict[str, Any]:
    """Storage -> remote host -> workstation, the path `mbo view` actually takes.

    Always the slowest of the three, and the only one that predicts how a
    remote dataset will feel to open.
    """
    rc, elapsed = _timed_retry(
        _ssh_base(host) + [
            f"dd if={_shq(path)} bs=1M skip={skip_mb} count={size_mb} status=none"
        ]
    )
    if rc != 0 or elapsed <= 0:
        return {"ok": False, "path": path}
    return {
        "ok": True,
        "path": path,
        "size_mb": size_mb,
        "throughput_mibs": size_mb / elapsed,
    }


def probe_metadata(host: str, path: str,
                   samples: int = _STAT_SAMPLES) -> dict[str, Any]:
    """Per-operation stat and directory-listing latency on the remote path.

    Browsing a folder is thousands of these, not one big read. On a
    high-latency mount this is what makes a file picker feel broken while
    the bandwidth numbers still look fine.
    """
    quoted = _shq(path)
    out = _ssh_capture(
        host,
        f"d=$(dirname {quoted}); "
        f"s=$(date +%s%N); "
        f"for i in $(seq {samples}); do stat {quoted} >/dev/null 2>&1; done; "
        f"e=$(date +%s%N); echo $(( (e-s)/1000000 )); "
        f's=$(date +%s%N); for i in $(seq 5); do ls -la "$d" >/dev/null 2>&1; done; '
        f"e=$(date +%s%N); echo $(( (e-s)/1000000 ))",
    )
    if not out:
        return {"ok": False, "path": path}
    nums = [ln.strip() for ln in out.splitlines() if ln.strip().isdigit()]
    if len(nums) < 2:
        return {"ok": False, "path": path}
    return {
        "ok": True,
        "path": path,
        "stat_ms": int(nums[0]) / samples,
        "listdir_ms": int(nums[1]) / 5,
    }


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def run_checks(host: str, path: str | None = None,
               size_mb: int = _DEFAULT_SIZE_MB, streams: int = 4,
               quick: bool = False, on_progress=None) -> dict[str, Any]:
    """Run the probe set and return one result dict.

    Probe failures are recorded in the result rather than raised: a partial
    report is more use than a traceback when the point is to find what broke.
    """
    def _note(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    results: dict[str, Any] = {
        "host": host,
        "hostname": resolve_hostname(host),
        "path": path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "local_link": local_link_mbps(),
    }

    _note("latency")
    results["latency"] = probe_latency(host)
    if not results["latency"].get("ok"):
        # every remaining probe rides the same SSH connection, so a dead link
        # is worth reporting once rather than as six separate timeouts
        results["reachable"] = False
        return results
    results["reachable"] = True

    _note("tcp connect")
    results["tcp_connect"] = probe_tcp_connect(host)

    _note(f"ssh throughput ({size_mb} MiB each way)")
    results["ssh"] = probe_ssh_throughput(host, size_mb)

    if not quick:
        _note(f"{streams} parallel streams")
        results["parallel"] = probe_parallel_streams(
            host, streams, max(size_mb // 2, 64)
        )

    if path:
        _note("remote mount")
        results["mount"] = probe_remote_mount(host, path)
        _note("metadata latency")
        results["metadata"] = probe_metadata(host, path)
        # read past the first GiB so a warm page cache does not flatter the result
        _note("remote storage read")
        results["remote_read"] = probe_remote_read(host, path, size_mb, skip_mb=1024)
        if not quick:
            _note("end-to-end read")
            results["end_to_end"] = probe_end_to_end(
                host, path, size_mb, skip_mb=2048
            )

    return results


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _mibs(value: float | None) -> str:
    if value is None:
        return "failed"
    return f"{value:,.0f} MiB/s ({value * 8 / 1000:.2f} Gbit/s)"


def _kib(blocks: str) -> str:
    """Df's raw 1024-byte block count -> human size. Passed through if odd."""
    try:
        value = float(blocks) * 1024
    except (TypeError, ValueError):
        return str(blocks)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if value < 1024 or unit == "PB":
            return f"{value:,.1f} {unit}"
        value /= 1024
    return str(blocks)


def _eta(mibs: float | None, size_mib: float) -> str:
    if not mibs or mibs <= 0:
        return "?"
    seconds = size_mib / mibs
    return f"{seconds:.0f}s" if seconds < 90 else f"{seconds / 60:.1f} min"


def format_report(results: dict[str, Any]) -> str:
    """Human-readable report over the dict from :func:`run_checks`."""
    lines: list[str] = []
    host = results["host"]
    hostname = results.get("hostname")
    target = f"{host} ({hostname})" if hostname and hostname != host else host
    lines.append(f"Network check: {target}")
    lines.append(f"  {results['timestamp']}")

    link = results.get("local_link")
    line_rate = link.get("mbps") if link else None
    if link:
        lines.append(
            f"  Local NIC: {link.get('description') or link.get('name')} "
            f"@ {link.get('link_speed')}"
        )

    if not results.get("reachable"):
        lines.append("")
        lines.append(f"UNREACHABLE - no response from {target}.")
        lines.append(f"  Check VPN/campus network, then: ssh -v {host}")
        return "\n".join(lines)

    lat = results.get("latency", {})
    if lat.get("ok"):
        note = {
            "tcp-handshake": " (tcp handshake to the ssh port)",
            "icmp": "",
            "ssh-exec": " (via ssh, includes key exchange - an upper bound)",
        }.get(lat["method"], "")
        lines.append("")
        lines.append(f"Latency{note}:")
        lines.append(
            f"  {lat['avg_ms']:.2f} ms avg   {lat['min_ms']:.2f} min / "
            f"{lat['max_ms']:.2f} max   jitter {lat['jitter_ms']:.2f} ms"
        )

    tcp = results.get("tcp_connect")
    if tcp and tcp.get("ok"):
        lines.append(
            f"  tcp connect to sshd: {tcp['avg_ms']:.2f} ms avg "
            f"({tcp['min_ms']:.2f} min / {tcp['max_ms']:.2f} max)"
        )

    ssh = results.get("ssh")
    if ssh:
        lines.append("")
        lines.append(f"SSH throughput ({ssh['size_mb']} MiB each way):")
        lines.append(f"  down  {_mibs(ssh.get('download_mibs'))}")
        lines.append(f"  up    {_mibs(ssh.get('upload_mibs'))}")
        down = ssh.get("download_mibs")
        if down and line_rate:
            pct = (down * 8 / 1000) / (line_rate / 1000) * 100
            lines.append(f"  = {pct:.0f}% of the {link['link_speed']} local line rate")

    par = results.get("parallel")
    if par and par.get("ok"):
        lines.append("")
        lines.append(f"Parallel ({par['streams']} streams):")
        lines.append(f"  {_mibs(par['aggregate_mibs'])} aggregate")

    mount = results.get("mount")
    read = results.get("remote_read")
    meta = results.get("metadata")
    # one section, printed whenever any storage probe produced something, so
    # a failed mount lookup cannot orphan the read/stat lines under no header
    if mount or read or meta:
        lines.append("")
        lines.append("Remote storage:")
        if mount and mount.get("ok"):
            lines.append(
                f"  {mount.get('mountpoint', mount['path'])}  "
                f"[{mount.get('fstype', '?')}]  {mount.get('source', '?')}"
            )
            if mount.get("size"):
                lines.append(
                    f"  {_kib(mount['size'])} total, "
                    f"{mount.get('used_pct', '?')} used"
                )
        elif mount:
            lines.append(f"  {results.get('path')}  [mount lookup failed]")

        if read and read.get("ok"):
            lines.append(f"  read (server-side)  {_mibs(read['read_mibs'])}")
        elif read:
            lines.append(
                f"  read (server-side)  failed: {read.get('reason', 'unknown')}"
            )

        if meta and meta.get("ok"):
            lines.append(
                f"  stat  {meta['stat_ms']:.1f} ms/op     "
                f"listdir  {meta['listdir_ms']:.1f} ms/op"
            )
        elif meta:
            lines.append("  stat/listdir  failed")

    e2e = results.get("end_to_end")
    if e2e and e2e.get("ok"):
        lines.append("")
        lines.append("End-to-end (storage -> server -> here):")
        lines.append(f"  {_mibs(e2e['throughput_mibs'])}")
        lines.append(
            f"  ~{_eta(e2e['throughput_mibs'], 6.2 * 1024)} "
            f"for a 6.2 GB session file"
        )

    verdict = _verdict(results, line_rate)
    if verdict:
        lines.append("")
        lines.extend(verdict)

    return "\n".join(lines)


def _verdict(results: dict[str, Any], line_rate: float | None) -> list[str]:
    """Name the bottleneck, since the raw numbers do not name it themselves."""
    lines = ["Assessment:"]

    ssh = results.get("ssh") or {}
    down = ssh.get("download_mibs")
    par = results.get("parallel") or {}
    agg = par.get("aggregate_mibs") if par.get("ok") else None
    e2e = results.get("end_to_end") or {}

    # judge saturation on the best throughput any probe reached, not on one
    # direction: run-to-run variance is several percent, and a single sample
    # sitting on the threshold flips the verdict for no physical reason
    observed = [v for v in (
        down,
        ssh.get("upload_mibs"),
        agg,
        e2e.get("throughput_mibs") if e2e.get("ok") else None,
    ) if v]
    peak = max(observed) if observed else None

    if peak and line_rate:
        frac = (peak * 8 / 1000) / (line_rate / 1000)
        if frac >= _LINK_SATURATED_FRAC:
            lines.append(
                f"  - Link is saturated (peak {peak:,.0f} MiB/s = {frac * 100:.0f}% "
                f"of line rate). The local NIC is the ceiling, not the campus "
                f"network."
            )
        elif frac >= _LINK_RATE_WARN_FRAC:
            lines.append(
                f"  - Peak {peak:,.0f} MiB/s is {frac * 100:.0f}% of line rate - "
                f"reasonable, with some headroom left."
            )
        else:
            lines.append(
                f"  - Peak {peak:,.0f} MiB/s is only {frac * 100:.0f}% of line "
                f"rate. Something upstream is limiting it (path, contention, or "
                f"duplex mismatch)."
            )

    if down and agg:
        if agg > down * _PARALLEL_GAIN_FRAC:
            lines.append(
                f"  - Parallel streams beat a single one ({agg:,.0f} vs {down:,.0f} "
                f"MiB/s): the link has headroom, so raise transfer concurrency."
            )
        else:
            lines.append(
                "  - Parallel streams do not beat a single one: the link is the "
                "limit, so more concurrency will not help."
            )

    meta = results.get("metadata") or {}
    if meta.get("ok"):
        if meta["listdir_ms"] > _SLOW_LISTDIR_MS:
            lines.append(
                f"  - Directory listings are slow ({meta['listdir_ms']:.0f} ms). "
                f"Browsing folders will feel worse than transfers do."
            )
        elif meta["stat_ms"] > _SLOW_STAT_MS:
            lines.append(
                f"  - Per-file metadata is slow ({meta['stat_ms']:.1f} ms/stat); "
                f"expect lag in file pickers over large folders."
            )

    read = results.get("remote_read") or {}
    if read.get("ok") and e2e.get("ok"):
        if read["read_mibs"] > e2e["throughput_mibs"] * _STORAGE_FASTER_FRAC:
            lines.append(
                f"  - Storage ({read['read_mibs']:,.0f} MiB/s) is faster than the "
                f"link to here ({e2e['throughput_mibs']:,.0f} MiB/s): the network "
                f"hop is the bottleneck, so process on the server where you can."
            )
        else:
            lines.append(
                "  - Storage read and end-to-end are comparable: the mount, not "
                "the network, sets the pace."
            )

    lat = results.get("latency") or {}
    slow_rtt = lat.get("ok") and lat.get("method") in ("icmp", "tcp-handshake")
    if slow_rtt and lat["avg_ms"] > _HIGH_RTT_MS:
        lines.append(
            f"  - {lat['avg_ms']:.0f} ms RTT is high for a campus link; interactive "
            f"work will feel sluggish regardless of bandwidth."
        )

    return lines if len(lines) > 1 else []

"""Host, hardware and Ollama environment capture.

Every experimental run records this. The reason is a specific incident during
development: a VS Code Remote-SSH session silently forwarded the Raspberry
Pi's Ollama port to the laptop's localhost, so benchmarks believed to be
running on the laptop were in fact executing on the Pi. The measurements
looked plausible. Only a physical observation, the Pi's fan spinning up
during a laptop command, revealed it.

That failure mode is invisible in results and fatal to a hardware comparison,
so the environment is now captured and fingerprinted rather than assumed.
``model_store_fingerprint`` in particular is a hash of the responding
server's model digests: if a run on the laptop reports the Pi's fingerprint,
the substitution is obvious in the results file.

Everything here degrades gracefully. A missing tool returns None rather than
raising, because a benchmark that refuses to run for want of a temperature
reading is worse than one that reports the temperature as unknown.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

CMD_TIMEOUT = 10


def _run(command: list[str]) -> str | None:
    """Run a command, returning stdout, or None if it is unavailable or fails."""
    if not shutil.which(command[0]):
        return None
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=CMD_TIMEOUT, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


# --- memory -----------------------------------------------------------------


def _memory_bytes() -> dict[str, int | None]:
    system = platform.system()
    if system == "Linux":
        try:
            values = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    values[key] = int(rest.split()[0]) * 1024
            return {"total": values.get("MemTotal"), "available": values.get("MemAvailable")}
        except (OSError, ValueError, IndexError):
            return {"total": None, "available": None}

    if system == "Windows":
        try:
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(MemoryStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
            return {"total": status.ullTotalPhys, "available": status.ullAvailPhys}
        except Exception:
            return {"total": None, "available": None}

    return {"total": None, "available": None}


# --- thermal and power ------------------------------------------------------


def cpu_temperature_c() -> float | None:
    """CPU temperature in Celsius, Linux only.

    Recorded per run because sustained inference on a passively cooled board
    can push it into thermal throttling, which changes the very latency the
    experiment is measuring.
    """
    for candidate in (
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/devices/virtual/thermal/thermal_zone0/temp",
    ):
        try:
            raw = int(Path(candidate).read_text().strip())
        except (OSError, ValueError):
            continue
        return raw / 1000.0 if raw > 1000 else float(raw)
    return None


def throttle_state() -> dict[str, Any] | None:
    """Raspberry Pi throttle flags, decoded.

    ``vcgencmd get_throttled`` returns a bitmask. The low bits report the
    current state, the high bits report whether the condition has occurred at
    any point since boot. A run that throttled is not directly comparable
    with one that did not, so this is recorded rather than inferred.
    """
    output = _run(["vcgencmd", "get_throttled"])
    if not output or "=" not in output:
        return None
    try:
        value = int(output.split("=")[1], 16)
    except (ValueError, IndexError):
        return None

    return {
        "raw": hex(value),
        "now": {
            "under_voltage": bool(value & 0x1),
            "arm_frequency_capped": bool(value & 0x2),
            "throttled": bool(value & 0x4),
            "soft_temp_limit": bool(value & 0x8),
        },
        "since_boot": {
            "under_voltage": bool(value & 0x10000),
            "arm_frequency_capped": bool(value & 0x20000),
            "throttled": bool(value & 0x40000),
            "soft_temp_limit": bool(value & 0x80000),
        },
    }


def power_profile() -> str | None:
    """Active power plan on Windows.

    Included because a laptop on the Balanced plan produced roughly a quarter
    of the throughput it produced on Best Performance. Without recording it,
    that variation would appear as unexplained noise between runs.
    """
    if platform.system() != "Windows":
        return None
    output = _run(["powercfg", "/getactivescheme"])
    if not output:
        return None
    match = re.search(r"\((.+?)\)\s*$", output)
    return match.group(1) if match else output


def gpu_info() -> list[dict[str, Any]]:
    """Discrete NVIDIA GPUs, if nvidia-smi is present."""
    output = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,temperature.gpu",
        "--format=csv,noheader,nounits",
    ])
    if not output:
        return []
    gpus = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpus.append({
                "name": parts[0],
                "memory_total_mb": int(parts[1]),
                "memory_used_mb": int(parts[2]),
                "temperature_c": int(parts[3]),
            })
        except ValueError:
            continue
    return gpus


# --- Ollama -----------------------------------------------------------------


def _get_json(url: str, timeout: int = 15) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_info(base_url: str) -> dict[str, Any]:
    """Version, model store fingerprint, and currently loaded models.

    ``model_store_fingerprint`` is the guard against endpoint substitution. It
    hashes the sorted digests of every model the responding server holds. Two
    machines with different model stores produce different fingerprints, so a
    tunnelled or misdirected endpoint shows up immediately in the results file
    instead of silently corrupting a hardware comparison.
    """
    info: dict[str, Any] = {"base_url": base_url, "reachable": False}

    try:
        info["version"] = _get_json(f"{base_url}/api/version").get("version")
        info["reachable"] = True
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError) as exc:
        info["error"] = str(exc)
        return info

    try:
        tags = _get_json(f"{base_url}/api/tags").get("models", [])
        models = sorted(
            (
                {
                    "name": m.get("name"),
                    "digest": (m.get("digest") or "")[:12],
                    "size": m.get("size"),
                }
                for m in tags
            ),
            key=lambda m: m["name"] or "",
        )
        info["models"] = models
        canonical = "|".join(f"{m['name']}:{m['digest']}" for m in models)
        info["model_store_fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        info["models"] = []
        info["model_store_fingerprint"] = None

    try:
        loaded = _get_json(f"{base_url}/api/ps").get("models", [])
        info["loaded"] = [
            {
                "name": m.get("name"),
                "size": m.get("size"),
                "size_vram": m.get("size_vram"),
                "context_length": m.get("context_length"),
            }
            for m in loaded
        ]
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        info["loaded"] = []

    return info


def gpu_offload_fraction(ollama: dict[str, Any]) -> float | None:
    """Fraction of the loaded model resident in VRAM, 0.0 to 1.0.

    Ollama reports total size and the portion in VRAM. The ratio is what the
    CLI displays as "100% GPU" or "80%/20% CPU/GPU". Recorded because a model
    that silently fell back to CPU produces latency figures that say nothing
    about the hardware they were supposedly measuring.
    """
    loaded = ollama.get("loaded") or []
    if not loaded:
        return None
    total = sum(m.get("size") or 0 for m in loaded)
    vram = sum(m.get("size_vram") or 0 for m in loaded)
    return round(vram / total, 4) if total else None


# --- assembly ---------------------------------------------------------------


def host_info() -> dict[str, Any]:
    """Everything about the machine that could plausibly affect a measurement."""
    import os

    memory = _memory_bytes()
    return {
        "hostname": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "memory_total_gb": round(memory["total"] / 2**30, 2) if memory["total"] else None,
        "memory_available_gb": round(memory["available"] / 2**30, 2) if memory["available"] else None,
        "cpu_temperature_c": cpu_temperature_c(),
        "throttle": throttle_state(),
        "power_profile": power_profile(),
        "gpus": gpu_info(),
    }


def environment(base_url: str) -> dict[str, Any]:
    """Full environment record for a run manifest."""
    ollama = ollama_info(base_url)
    return {
        "host": host_info(),
        "ollama": ollama,
        "gpu_offload_fraction": gpu_offload_fraction(ollama),
    }


# --- source provenance ------------------------------------------------------


def git_commit(repo_root: Path | None = None) -> dict[str, Any]:
    """Current commit, branch and working-tree cleanliness.

    Recorded per run so a results file identifies the exact source that
    produced it. ``dirty`` matters as much as the SHA: a run made with
    uncommitted changes is not reproducible from the commit alone, and that
    should be visible rather than assumed away.
    """
    root = repo_root or Path(__file__).resolve().parents[3]
    base = ["git", "-C", str(root)]
    sha = _run([*base, "rev-parse", "HEAD"])
    if sha is None:
        return {"available": False}
    status = _run([*base, "status", "--porcelain"])
    return {
        "available": True,
        "commit": sha,
        "commit_short": sha[:12],
        "branch": _run([*base, "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        # Porcelain format is two status characters, a space, then the path.
        "dirty_files": [line[2:].strip() for line in status.splitlines()] if status else [],
    }

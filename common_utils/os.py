import ctypes.util
import faulthandler
import logging
import os
import re
import signal
from datetime import datetime

import psutil
from tabulate import tabulate

_libc = ctypes.CDLL(ctypes.util.find_library('c'))
_PR_SET_PDEATHSIG = 1
LOGGER = logging.getLogger(__name__)


def set_death_signal(signal):
    """Set process flag to send signal on parent death"""
    LOGGER.info(f'Set _PR_SET_PDEATHSIG flag to {signal} for {os.getpid()}')
    _libc.prctl(_PR_SET_PDEATHSIG, signal)


def set_death_signal_kill():
    set_death_signal(signal.SIGKILL)


def set_death_signal_term():
    set_death_signal(signal.SIGTERM)


def set_death_signal_int():
    set_death_signal(signal.SIGINT)


def enable_faulthandlers():
    """Dump stack trace on error signals + SIGUSR1"""
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1)


def terminate_process_tree_if_exists(pid, *, termination_timeout=3, kill_timeout=60):
    try:
        proc = psutil.Process(pid)
        procs = proc.children(recursive=True) + [proc, ]

        # send SIGTERM
        for p in procs:
            # LOGGER.debug(f'Terminating {p.pid}/{p.name()} current status: {p.status()}')
            p.terminate()
        gone, alive = psutil.wait_procs(procs, timeout=termination_timeout)

        if alive:
            # send SIGKILL
            for p in alive:
                LOGGER.debug(f'Still alive {p.pid} - then killing; status: {p.status()}')
                p.kill()
            gone2, alive = psutil.wait_procs(alive, timeout=kill_timeout)
            gone.extend(gone2)

        alive = list(filter(lambda a: a.status() != psutil.STATUS_ZOMBIE, alive))

        return alive
    except psutil.NoSuchProcess:
        # Cant atomically check-process-exists-and-kill-it. Between check and trying-to-kill process might be killed
        # from other place.
        # Need to kill it and recover if it failes
        LOGGER.warn("Process already killed from other source")
        return []

def dump_subprocess_info(pid=None, process_rexp=None):
    def extract_info(p):
        ct = datetime.fromtimestamp(p.create_time()).strftime("%Y-%m-%d %H:%M:%S")
        mem_mb = p.memory_info().rss / (1024 * 1024)
        return [p.pid, p.ppid(), ct, p.name(), p.cpu_percent(), mem_mb, p.memory_percent(), p.status()]

    headers = ['PID', 'PPID', 'created_at', 'name', 'cpu_percent', 'mem[MB]', 'mem_percent', 'status']
    process = psutil.Process(pid=pid)

    processes = [process, ] + process.children(recursive=True)
    if process_rexp:
        processes = [p for p in processes if re.match(process_rexp, p.name())]
    info = [extract_info(p) for p in processes]
    print(tabulate(info, headers=headers))


def get_open_port():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port

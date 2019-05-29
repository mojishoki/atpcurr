import os
import shutil
import signal
import tempfile
import time
from datetime import datetime
from functools import wraps
from pathlib import Path


class GracefulInterruptHandler(object):

    def __init__(self, sig=signal.SIGINT):
        self.sig = sig

    def __enter__(self):
        self.interrupted = False
        self.released = False

        self.original_handler = signal.getsignal(self.sig)

        def handler(signum, frame):
            self.release()
            self.interrupted = True

        signal.signal(self.sig, handler)

        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        if self.released:
            return False

        signal.signal(self.sig, self.original_handler)

        self.released = True

        return True


def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        print(f"fun: {f.__name__}, args: [{args}, {kwargs}] took: {te - ts} sec")
        return result

    return wrap


class TempDir(object):

    def __init__(self, *args, **kwargs):
        self.dir = Path(tempfile.mkdtemp(*args, **kwargs))

    def __repr__(self):
        return "<{} {!r}>".format(self.__class__.__name__, self.dir.as_posix())

    def __enter__(self):
        return self.dir

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def __exit__(self, exc, value, tb):
        self.cleanup()

    def __del__(self):
        self.cleanup()


DEFAULT_LOG_DIR_REL = '../logs'


def get_log_dir(log_subdir, append_dt=True):
    log_dir = Path(os.environ.get('LOG_DIR', DEFAULT_LOG_DIR_REL)) / 'experiments' / log_subdir
    if append_dt:
        dt = datetime.now().strftime("%Y%m%d_%H%M%S-%f")
        log_dir = log_dir / dt
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir.absolute().as_posix()



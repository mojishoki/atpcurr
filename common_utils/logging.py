# -*- coding: utf-8 -*-

import logging
import os
import re
import shutil
import typing

LOGGING_FORMAT = '%(asctime)-15s %(process)d %(threadName)s %(levelname)-8s %(name)s %(message)s'


class LogFilter(logging.Filter):
    class FilterEntry(typing.NamedTuple):
        name_regexp: str
        msg_regexp: str

        def match(self, record: logging.LogRecord):
            match_name = bool(re.match(self.name_regexp, record.name))
            match_regexp = bool(re.match(self.msg_regexp, record.getMessage()))
            return match_name and match_regexp

    def filter(self, record: logging.LogRecord):
        excluded = [
            LogFilter.FilterEntry(r'neptune\.generated.*', r'.*'),
            LogFilter.FilterEntry(r'neptune\.internal.*', r'.*'),
            LogFilter.FilterEntry(r'urllib3\.connectionpool', r'.*'),
            LogFilter.FilterEntry(r'root', r'\(localhost.*\).*connect.*'),
            LogFilter.FilterEntry(r'matplotlib\.axes.*', r'.*'),
            LogFilter.FilterEntry(r'matplotlib\.font_manager.*', r'.*'),
            LogFilter.FilterEntry(r'PIL\.PngImagePlugin.*', r'.*'),
        ]
        shall_be_excluded = record.levelno == logging.DEBUG and any([entry.match(record) for entry in excluded])
        return not shall_be_excluded


class EpochRollingDirectories:

    def __init__(self, root: str, epochs_to_keep: int=5):
        self._root = root
        self._epochs_to_keep = epochs_to_keep
        self._logged_epochs = []
        self._directories_rolling_window = []

    def create_new_and_delete_old(self, epoch_no: int):
        if len(self._logged_epochs) > 0:
            assert epoch_no > self._logged_epochs[0], "Epoch no must increase"

        dir = self._dir_for_epoch(epoch_no)
        os.makedirs(dir, exist_ok=True)

        self._logged_epochs.insert(0, epoch_no)

        if len(self._logged_epochs) > self._epochs_to_keep:
            dir_to_remove = self._dir_for_epoch(self._logged_epochs.pop())
            shutil.rmtree(dir_to_remove, ignore_errors=True)

        return dir

    def _dir_for_epoch(self, epoch_no):
        return os.path.join(self._root, str(epoch_no).zfill(7))
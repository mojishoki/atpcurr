# -*- coding: utf-8 -*-
import json
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import requests

from baselines.logger import KVWriter

_PROJECT_ID = ''
_NEPTUNE_TOKEN_PATH = ''


def try_get_short_id_of_current(neptune_ctx):
    # Using unsupported developer APIs. This might break after updating neptune-cli
    try:
        return neptune_ctx._experiment._services._job_api_service.get_experiment(neptune_ctx._experiment.id).short_id
    except:
        print("Could not obtain neptune short_id. Possible reasons: [offline context, upgraded not-compatible neptune-cli]")
        return None


def get_local_access_token():
    # refresh token; raise exception if could not authorize
    subprocess.run(['neptune', 'experiment', 'list'], check=True)
    access_token = json.load(Path(_NEPTUNE_TOKEN_PATH).expanduser().open())['access_token']
    return access_token


class NeptuneAPI(object):
    TOKEN_REFRESH_INTERVAL_S = 60

    def __init__(self, *, project_id=_PROJECT_ID):
        self._project_id = project_id
        self._cache_dir = tempfile.mkdtemp(prefix='neptune_channels')

        self._token_refreshed_at = None
        self._token = None

    @property
    def cache_dir(self):
        return self._cache_dir

    @property
    def access_token(self):
        if self._token_refreshed_at is None or time.time() - self._token_refreshed_at > self.TOKEN_REFRESH_INTERVAL_S:
            self._token = get_local_access_token()
            self._token_refreshed_at = time.time()
        return self._token

    def get_experiments_raw(self, *, tags=None):
        url = rf'https://app.neptune.ml/api/backend/v1/experiments?' \
            rf'trashed=false&' \
            rf'projectId={self._project_id}&' \
            rf'access_token={self.access_token}'
        for tag in (tags or []):
            url += f"&tags={tag}"

        response = requests.get(url)
        response.raise_for_status()

        return response.json()['entries']

    def get_experiments(self, *, tags=None):
        return [NeptuneExperiment(experiment=e, api=self) for e in self.get_experiments_raw(tags=tags)]


class NeptuneExperiment(object):

    def __init__(self, *, api: NeptuneAPI, experiment: dict):
        """pass experiment dict obtained from API"""
        self._api = api
        self._experiment = experiment
        # check no duplicated channel names
        assert len([c['channelName'] for c in self._experiment['channelsLastValues']]) == \
               len({c['channelName'] for c in self._experiment['channelsLastValues']})
        self._experiment['channels'] = {c['channelName']: c for c in self._experiment['channelsLastValues']}
        self._experiment['props'] = {p['key']: p['value'] for p in self._experiment['properties']}

    @property
    def properties(self):
        return self._experiment['props']

    @property
    def channels(self):
        return self._experiment['channels']

    @property
    def short_id(self):
        return self._experiment['shortId']

    @property
    def tags(self):
        return self._experiment['tags']

    @property
    def state(self):
        return self._experiment['state']

    @property
    def created_at(self):
        import pandas as pd
        return pd.to_datetime(self._experiment['timeOfCreation'])

    @property
    def running_time(self):
        import pandas as pd
        return pd.to_timedelta(int(self._experiment['runningTime']), unit='ms')

    def _csv_url_for_channel(self, channel_name):
        experiment_id = self._experiment['id']
        token = self._api.access_token

        url = rf"https://app.neptune.ml/api/backend/v1/experiments/{ experiment_id }?access_token={ token }"
        response = requests.get(url)
        response.raise_for_status()
        experiment = response.json()
        channels = experiment['channels']

        channels = [c for c in channels if c['name'] == channel_name]
        if not channels:
            raise KeyError(f"Could not find {channel_name} for experiment {self._experiment['shortId']}")
        channel = channels[0]

        return rf'https://app.neptune.ml/api/backend/v1/experiments/{ experiment_id }/channel/{ channel["id"] }/csv?' \
            rf'access_token={token}'

    def get_channel_data(self, channel_name):
        import pandas as pd
        channel = self.channels[channel_name]
        cache_path = Path(self._api.cache_dir) / f"{ channel['channelId'] }_{ channel['x'] }.csv"
        if not cache_path.exists():
            with cache_path.open('w') as cache_file:
                csv_url = self._csv_url_for_channel(channel_name=channel_name)
                data = requests.get(csv_url).text
                cache_file.write(data)
        return pd.read_csv(cache_path, index_col=0, header=None)


class Neptune2OutputFormat(KVWriter):

    def __init__(self, neptune_ctx):
        self.neptune_ctx = neptune_ctx

    def writekvs(self, kvs: dict):
        x = kvs['update_no']
        for k, v in kvs.items():
            self.neptune_ctx.channel_send(k, x=x, y=v)

    def close(self):
        # TODO
        pass


def set_neptune_properties_for_experiment(neptune_ctx, args, num_nodes=None, log_dir: str = None):
    if num_nodes is not None:
        neptune_ctx.properties['SLURM_JOB_NUM_NODES'] = num_nodes
    neptune_ctx.properties['SLURM_JOBID'] = os.getenv('SLURM_JOBID', 'None')
    neptune_ctx.properties['SHA'] = os.getenv('SHA', 'None')
    neptune_ctx.properties['hostname'] = socket.gethostname()

    if log_dir is not None:
        neptune_ctx.properties['log_dir'] = os.path.abspath(log_dir)

    for key, value in vars(args).items():
        neptune_ctx.properties[key] = value

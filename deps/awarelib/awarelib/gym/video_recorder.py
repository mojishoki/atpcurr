import os

from gym import Wrapper

from gym.wrappers.monitoring import video_recorder
from pathlib2 import Path

from awarelib.logger import logger

class RecordVideoTrigger(object):
    def __call__(self, step_id, episode_id):
        raise NotImplementedError


class AlwaysTrueRecordVideoTrigger(RecordVideoTrigger):
    def __call__(self, step_id, episode_id):
        return True

class VideoRecorderWrapper(Wrapper):
    """
    Wrap Env to record rendered image as mp4 video.
    NOTE(maciek): this is almost entirely copied from vec_video_recorder.py from baselines
    """

    def __init__(self, env,
                 directory='/tmp/gym_videos/',
                 record_video_trigger=AlwaysTrueRecordVideoTrigger(),
                 video_length=2000):

        super(VideoRecorderWrapper, self).__init__(env)

        self.record_video_trigger = record_video_trigger
        self.video_recorder = None

        self.directory = os.path.abspath(directory)
        if not Path(self.directory).exists():
            Path(self.directory).mkdir(parents=True)

        self.file_prefix = "env"
        self.file_infix = '{}'.format(os.getpid())
        self.step_id = 0
        self.episode_id = -1
        self.video_length = video_length

        self.recording = False
        self.recorded_frames = 0

    def reset(self):
        self.episode_id += 1
        obs = self.env.reset()

        self.close_video_recorder()

        return obs

    def start_video_recorder(self):
        self.close_video_recorder()

        base_path = os.path.join(self.directory, '{}.video.{}.video_{:06}_{:010}'.format(
            self.file_prefix, self.file_infix, self.episode_id, self.step_id))
        self.video_recorder = video_recorder.VideoRecorder(
                env=self.env,
                base_path=base_path,
                metadata={'step_id': self.step_id, 'epsiode_id': self.episode_id}
                )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

    def _video_enabled(self):
        return self.record_video_trigger(step_id=self.step_id, episode_id=self.episode_id)

    def step(self, action):
        ob, rew, done, info = self.env.step(action)

        self.step_id += 1
        if self.recording:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if self.recorded_frames > self.video_length:
                logger.info("Saving video to ", self.video_recorder.path)
                self.close_video_recorder()
        elif self._video_enabled():
            self.start_video_recorder()

        return ob, rew, done, info

    def close_video_recorder(self):
        if self.recording:
            self.video_recorder.close()
        self.recording = False
        self.recorded_frames = 0

    def close(self):
        super(VideoRecorderWrapper, self).close()
        self.close_video_recorder()

    def __del__(self):
        self.close()

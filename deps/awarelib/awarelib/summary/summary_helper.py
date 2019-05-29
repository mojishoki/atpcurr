import sys
from collections import defaultdict
import numpy as np

import tensorflow as tf


def create_simple_summary(tag, y):
    return tf.Summary(value=[tf.Summary.Value(tag=tag,
                                              simple_value=y)])


def add_simple_summary(tag, y, summary_writer, global_step=None):
    summary = create_simple_summary(tag, y)
    summary_writer.add_summary(summary, global_step=global_step)


def mean_aggregator(l):
    return np.mean(l)


class SummaryHelper(object):
    def __init__(self, summary_writer):
        self.summary_writer = summary_writer
        self.channels = defaultdict(list)

    def add_simple_summary(self, tag, y, freq=20, global_step=None, aggregator=mean_aggregator,
                           force_neptune=False, neptune_ctx=None):
        self.channels[tag].append(y)
        if len(self.channels[tag]) >= freq:
            aggregated_y = aggregator(self.channels[tag])

            add_simple_summary(tag=tag,
                               global_step=global_step,
                               y=aggregated_y,
                               summary_writer=self.summary_writer)
            if force_neptune:
                assert neptune_ctx is not None
                print('Will send value, {}, {}, {}'.format(tag, global_step, y))
                neptune_ctx.channel_send(tag,
                                         x=global_step,
                                         y=y)
            # self.summary_writer.flush()
            self.channels[tag] = []

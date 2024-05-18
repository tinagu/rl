import os

import numpy as np
import torch
from tensorboardX import SummaryWriter
from collections import defaultdict
from datetime import datetime
import abc
import csv


class Writer(object, metaclass=abc.ABCMeta):

    def __init__(self):
        pass

    @abc.abstractmethod
    def record(self, k, v, prefix=None):
        pass

    def record_dict(self, key: str, value):
        pass

    @abc.abstractmethod
    def dump(self, epoch_idx=0):
        pass


class StdoutWriter(Writer):

    def __init__(self):
        super(StdoutWriter, self).__init__()

        self.tbl_dict = {}

    def record(self, k, v, prefix=None):
        if prefix:
            k = f"{prefix}/{k}"
        if k not in self.tbl_dict:
            self.tbl_dict[k] = []
        self.tbl_dict[k].append(v)

    def record_dict(self, dct, prefix=None):
        for k, v in dct.items():
            if prefix:
                k = f"{prefix}/{k}"
            self.tbl_dict[k] = v

    def dump(self, epoch_idx=0):
        stats_dict = {}
        for k, v in self.tbl_dict.items():
            try:
                if isinstance(v, list):
                    stats_dict[f"{k}_mean"] = np.mean(v)
                    stats_dict[f"{k}_max"] = np.max(v)
                    stats_dict[f"{k}_min"] = np.min(v)
                    stats_dict[f"{k}_count"] = len(v)
                else:
                    stats_dict[k] = v
            except Exception as e:
                print(k)
                raise e
        for k, v in stats_dict.items():
            print(epoch_idx, k, v)
        print("\n")
        self.tbl_dict.clear()


class TableWriter(Writer):

    def __init__(self, output_dir):
        super(TableWriter, self).__init__()

        self.output_dir = output_dir
        self.csv_filename = f"{self.output_dir}/epochs.csv"
        self.tbl_dict = {}
        self.headers = None
        self.new = True

    def dump(self, epoch_idx=0):
        stats_dict = {}
        for k, v in self.tbl_dict.items():
            if isinstance(v, list):
                stats_dict[f"{k}_mean"] = np.mean(v)
                stats_dict[f"{k}_max"] = np.max(v)
                stats_dict[f"{k}_min"] = np.min(v)
                stats_dict[f"{k}_count"] = len(v)
            else:
                stats_dict[k] = v
        self.headers = list(stats_dict.keys())

        with open(self.csv_filename, "a+") as o_f:
            if self.new:
                o_f.write(','.join(self.headers))
                self.new = False

            writer = csv.DictWriter(o_f, fieldnames=list(stats_dict.keys()))
            writer.writerow(stats_dict)
        self.tbl_dict.clear()

    def record(self, k, v, prefix=None):
        if prefix:
            k = f"{prefix}/{k}"
        if k not in self.tbl_dict:
            self.tbl_dict[k] = []
        self.tbl_dict[k].append(v)

    def record_dict(self, dct, prefix=None):
        for k, v in dct.items():
            if prefix:
                k = f"{prefix}/{k}"
            self.tbl_dict[k] = v


class TensorboardWriter(Writer):

    def __init__(self, output_dir):
        super(TensorboardWriter, self).__init__()

        self.output_dir = output_dir
        self.output_dir = os.path.join(self.output_dir, "tb")
        os.makedirs(self.output_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.output_dir)
        self.name_to_value = defaultdict(float)

    def write(self, kv, epoch_idx) -> None:
        for k, v in kv.items():
            if isinstance(v, np.ScalarType):
                if isinstance(v, str):
                    self.writer.add_text(k, v, epoch_idx)
                else:
                    self.writer.add_scalar(k, v, epoch_idx)

            if isinstance(v, torch.Tensor):
                self.writer.add_histogram(k, v, epoch_idx)

        self.writer.flush()

    def record(self, k, v, prefix=None):
        self.name_to_value[k] = v

    def record_dict(self, dct, prefix=None):
        pass

    def dump(self, epoch_idx=0):
        self.write(self.name_to_value, epoch_idx)
        self.writer.flush()

        self.name_to_value.clear()


class Logger(Writer):

    def __init__(self, configs=None):
        super(Logger, self).__init__()

        self._loggers = []
        self.output_dir = None
        self.exp_name = None
        if configs:
            self.setup(configs)

    def setup(self, configs):
        self.output_dir = configs.get("log_dir") or "./logdir"
        self.exp_name = configs.get("exp_name")
        if self.exp_name:
            self.output_dir += f"/{self.exp_name}"
        timestamp = datetime.now().strftime("%y%m%d-%H%M%S%f")
        self.output_dir += f"/{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        self._loggers += [StdoutWriter(),
                          TableWriter(self.output_dir),
                          TensorboardWriter(self.output_dir)]

    def record(self, k, v, prefix=None):
        for _logger in self._loggers:
            _logger.record(k, v, prefix)

    def record_dict(self, k, v, prefix=None):
        for _logger in self._loggers:
            _logger.record(k, v, prefix)

    def dump(self, epoch_idx=0):
        for _logger in self._loggers:
            _logger.dump(epoch_idx)


logger = Logger()

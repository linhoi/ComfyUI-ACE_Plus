# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import importlib
import os
import sys
from datetime import datetime
sys.dont_write_bytecode = True
from scepter.modules.solver.registry import SOLVERS
from scepter.modules.utils.config import Config
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS
from scepter.modules.utils.logger import get_logger

if os.path.exists('__init__.py'):
    package_name = 'scepter_ext'
    spec = importlib.util.spec_from_file_location(package_name, '__init__.py')
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)

def run_task(cfg):
    std_logger = get_logger(name='scepter')
    solver = SOLVERS.build(cfg.SOLVER, logger=std_logger)
    solver.set_up_pre()
    solver.set_up()
    if we.rank == 0:
        FS.put_object_from_local_file(cfg.args.cfg_file, os.path.join(solver.work_dir, "train.yaml"))
    if cfg.args.stage == "train":
        solver.solve()
    elif cfg.args.stage == "eval":
        solver.run_eval()


def update_config(cfg):
    if hasattr(cfg.args, 'learning_rate') and cfg.args.learning_rate:
        print(
            f'learning_rate change from {cfg.SOLVER.OPTIMIZER.LEARNING_RATE} to {cfg.args.learning_rate}'
        )
        cfg.SOLVER.OPTIMIZER.LEARNING_RATE = float(cfg.args.learning_rate)
    if hasattr(cfg.args, 'max_steps') and cfg.args.max_steps:
        print(
            f'max_steps change from {cfg.SOLVER.MAX_STEPS} to {cfg.args.max_steps}'
        )
        cfg.SOLVER.MAX_STEPS = int(cfg.args.max_steps)
    cfg.SOLVER.WORK_DIR = os.path.join(cfg.SOLVER.WORK_DIR, "{0:%Y%m%d%H%M%S}".format(datetime.now()))
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argparser for Scepter:\n')
    parser.add_argument(
        "--stage",
        dest="stage",
        help="Running stage!",
        default="train",
        choices=["train", "eval"]
    )
    parser.add_argument('--learning_rate',
                        dest='learning_rate',
                        help='The learning rate for our network!',
                        default=None)
    parser.add_argument('--max_steps',
                        dest='max_steps',
                        help='The max steps for training!',
                        default=None)

    cfg = Config(load=True, parser_ins=parser)
    cfg = update_config(cfg)
    we.init_env(cfg, logger=None, fn=run_task)

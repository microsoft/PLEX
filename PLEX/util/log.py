from datetime import datetime
from pathlib import Path
import json
import sys
import random
import string
import wandb


class Log:
    def __init__(self, log_dir, filename='log.txt', flush=True):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir/filename
        self.file = open(self.path, 'w')
        self.flush = flush

    def write(self, message, end='\n'):
        now_str = datetime.now().strftime('%H:%M:%S')
        message = f'[{now_str}] ' + message
        for f in [sys.stdout, self.file]:
            print(message, end=end, file=f, flush=self.flush)

    def __call__(self, *args, **kwargs):
        self.write(*args, **kwargs)

    def close(self):
        self.file.close()


def setup_logging(args):
    log_dir = Path(args['log_dir']).expanduser()
    if not log_dir.is_dir():
        print(f'Creating log dir {log_dir}')
        log_dir.mkdir(parents=True)

    now_str = datetime.now().strftime('%m-%d-%y_%H.%M.%S')
    log_id = args.get('log_id', ''.join(random.choice(string.ascii_lowercase) for _ in range(4)))
    run_log_dir = log_dir/f'{now_str}_{log_id}'
    run_log_dir.mkdir()
    log = Log(run_log_dir)
    log(f'Log path: {log.path}')

    config_path = run_log_dir/'config.json'
    log(f'Dumping config to {config_path}')
    config_path.write_text(json.dumps(args))

    return log


def setup_wandb_logging(group_name, cmdline_args):
    exp_prefix = f'{group_name}_{random.randint(int(1e5), int(1e6) - 1)}'
    wandb.init(
        name=exp_prefix,
        group=group_name,
        project='PLEX',
        config=cmdline_args
    )
    # wandb.watch(model)  # wandb has some bug


import logging
import os
import pathlib
import subprocess
import sys
from pathlib import Path

import mlflow
from hydra import main as hydra_main
from hydra.core.hydra_config import HydraConfig
from mlflow import log_param

# jobs_run = 0


def main(*args, **kwargs):
    r"""
    The function returned will be a decorator that wraps a function to achieve
    the following functionality:
    1) if -m or --multi-run are found amid the command-line arguments, hydra is
        used to set up the multiple runs---each in a separate directory
        including a "run.sh" file that is suitable for "running" with bash,
        qsub, cat, ls, etc. The script will use hydra to run only that run
        (strips the -m arguments and points to the stored config without the
        remaining arguments), in the given folder. The script also creates
        .started and .finished files before and after the enclosed python script
        is run respectively.
    2) otherwise, the single job is run using hydra, setting up an mlflow run
       and logging the parameters to it.


    command-line-params:
    mlruns: this is where the mlruns will be saved (likely shared across may experiments)
    exp.run_with: what program will be used on the generated .sh file

    """

    script = str(pathlib.Path(sys.argv[0]).resolve())

    def wrap(f):
        def wrapped(cfg):
            try:
                mlruns = cfg['exp']['mlruns']
            except KeyError:
                mlruns = 'mlruns'
            mlruns_path = Path(mlruns).resolve()
            mlruns_path.mkdir(parents=True, exist_ok=True)
            logging.getLogger('alembic.runtime.migration').disabled = True
            mlflow.set_tracking_uri(f'sqlite:///{mlruns_path}/mlruns.db')

            # mlflow.set_tracking_uri(f'file://{mlruns}')
            mlflow.start_run()
            for k, v in cfg.items():
                log_param(k, v)
            f(cfg)
            mlflow.end_run()
        wrapped.__module__ = '__main__'
        return wrapped

    def make_script(cfg):
        # global jobs_run
        # jobs_run += 1
        cwd = os.getcwd()
        python = sys.executable
        # TODO: restarting??
        run_script = os.path.join(os.getcwd(), 'run.sh')
        try:
            run_with = cfg['exp']['run_with']
        except KeyError:
            run_with = 'bash'
        try:
            mlruns = cfg['exp']['mlruns']
        except KeyError:
            mlruns = HydraConfig.get()['sweep']['dir']
        with open(run_script, 'w') as f:
            print(f"""
#!/usr/bin/env bash

# (See qsub section for explanation on these flags.)
#$ -N {HydraConfig.get().job.name}
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -m e
#$ -wd {cwd}

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l ram_free=1G,mem_free=1G,gpu=1,hostname=b1[123456789]|c0*|c1[123456789]

# Submit to GPU queue
#$ -q g.q

echo "writing results to {cwd}..."
cd {cwd}
touch {cwd}/.started
getgpu=/home/gqin2/scripts/acquire-gpu
if [ -f "$getgpu" ]; then
    # Assign a free-GPU to your program (make sure -n matches the requested number of GPUs above)
    source /home/gqin2/scripts/acquire-gpu
fi
{python} {script} -cd {os.getcwd()}/.hydra --config-name config
touch {cwd}/.finished
""", file=f)
        subprocess.run([run_with, run_script])
    make_script.__module__ = '__main__'

    def wrapped(f):
        if '-m' in sys.argv or '--multirun' in sys.argv:
            return hydra_main(*args, **kwargs)(make_script)
        else:
            return hydra_main(*args, **kwargs)(wrap(f))
    return wrapped

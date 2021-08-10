
import os
import pathlib
import subprocess
import sys

from hydra import main as hydra_main
# jobs_run = 0
from hydra.core.hydra_config import HydraConfig


def main(*args, **kwargs):
    script = str(pathlib.Path(sys.argv[0]).resolve())

    def no_op(cfg):
        # global jobs_run
        # jobs_run += 1
        cwd = os.getcwd()
        python = sys.executable
        # args = ' '.join([arg for arg in sys.argv[1:] if arg.lower() not in [
        #     '-m', '--multirun']])
        # TODO: restarting
        run_script = os.path.join(os.getcwd(), 'run.sh')
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

# Assign a free-GPU to your program (make sure -n matches the requested number of GPUs above)
touch {cwd}/.started
#source /home/gqin2/scripts/acquire-gpu
{python} {script} -cd {os.getcwd()}/.hydra --config-name config
touch {cwd}/.finished
""", file=f)
        subprocess.run([cfg['job']['run_with'], run_script])
    no_op.__module__ = '__main__'

    def wrapped(f):
        if '-m' in sys.argv or '--multirun' in sys.argv:
            return hydra_main(*args, **kwargs)(no_op)
        else:
            return hydra_main(*args, **kwargs)(f)
    return wrapped

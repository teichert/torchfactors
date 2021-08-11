# Allows multirun launches by generated a script and then running it.
# Maybe I should have done this with a launcher plugin, but this seemed easier
# at the time.

import logging
import os
import pathlib
import subprocess
import sys
import warnings
from pathlib import Path

from hydra import main as hydra_main
from hydra.core.hydra_config import HydraConfig
from mlflow import log_param  # type: ignore


def main(*args, **kwargs):
    r"""
    The function returned will be a decorator that wraps a function to achieve
    the following functionality:
    1) If not in multi-run mode, then hydra normally would run the function with
        the provided argument configuration.  This wrapper additionally sets up
        a mlflow run that automatically has the parameter values logged and
        allows reporting of other results and artifacts.  The path to the mlflow
        results should be given by `exp.mlruns_path` (or mlruns.db will be used
        as default). The type should be given as file, sqlite, or http, or https
        via `exp.mlruns_type`.

    2) If -m or --multi-run are found amid the command-line arguments, hydra
        normally would sweep over the defined settings, creating a folder
        corresponding to each and then running the wrapped function within the
        respective folder with the appropriate argument configuration. This
        wrapper instead instructs hydra to create the folders but rather than
        running the function directly, it creates a custom script ("run.sh") for
        each folder which is then run using some specified application. The
        application to use for running the generated script should be specified
        via `exp.run_with` (default is "bash" other reasonable options might be
        "ls" or "cat" for dry-runs).  The contents of the script file should be
        specified via `exp.sub.script` which will be formatted with python
        string formatting while providing the following four variables (which
        can be used within curly braces [without a preceding $ since that would
        indicate that hydra should exapand the variable]):
        - {python} will be replaced with the path to the python interpreter
          currently being used
        - {script} will be replaced with the path to the python script being run
          (i.e. the one that is producing these files)
        - {cwd} will be replaced with the folder in which the script will be
          created
        - {hydra} is the HydraConfig (members can be accessed with e.g.
          {hydra.job.name} or {hydra.job.id}.

        If no `exp.sub.script` is defined, then a basic script will be created
        that simply runs the python file with the appropriate generated config.
    """

    running_script = str(pathlib.Path(sys.argv[0]).resolve())

    def run_with_mlflow(f):
        def run_with_config(cfg):
            import mlflow
            try:
                mlruns_root = cfg['exp']['mlruns_path']
            except KeyError:
                mlruns_root = 'mlruns.db'
            try:
                mlruns_type = cfg['exp']['mlruns_type']
            except KeyError:
                mlruns_type = 'sqlite'
            mlruns_path = Path(mlruns_root).resolve()
            mlruns_path.parent.mkdir(parents=True, exist_ok=True)
            # disabling becuase sqlite backend leads to many logger info messages and
            # switching log-level doesn't work
            logging.getLogger('alembic.runtime.migration').disabled = True
            mlruns_uri = f'{mlruns_type}:///{mlruns_path}'
            logging.getLogger(__name__).info(f'mlruns tracked at: {mlruns_uri}')
            mlflow.set_tracking_uri(mlruns_uri)
            with mlflow.start_run():
                for k, v in cfg.items():
                    if isinstance(v, (float, str, int, bool)):
                        log_param(k, v)
                f(cfg)
        run_with_config.__module__ = '__main__'
        return run_with_config

    def generate_script_with_config(cfg):
        try:
            run_with = cfg['exp']['run_with']
        except KeyError:
            run_with = 'bash'
        try:
            script_str = cfg['exp']['sub']['script']
        except KeyError:
            warnings.warn("no exp.sub.script given, using simple builtin")
            script_str = "{python} {script} -cd {cwd}/.hydra --config-name config"

        generated_script = os.path.join(os.getcwd(), 'run.sh')
        with open(generated_script, 'w') as f:
            logging.info(f"creating run script: {generated_script}")
            formatted = script_str.format(
                python=sys.executable,
                script=running_script,
                cwd=os.getcwd(),
                hydra=HydraConfig.get())
            print(formatted, file=f)
        subprocess.run([run_with, generated_script])

    generate_script_with_config.__module__ = '__main__'

    def run_with_myhydra(f):
        if '-m' in sys.argv or '--multirun' in sys.argv:
            return hydra_main(*args, **kwargs)(generate_script_with_config)
        else:
            return hydra_main(*args, **kwargs)(run_with_mlflow(f))
    return run_with_myhydra

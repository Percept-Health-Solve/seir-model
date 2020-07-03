from seir.argparser import DataClassArgumentParser
from seir.cli import OdeParamCLI, LockdownCLI
from seir.ode_parameters import OdeParams, LockdownParams, MetaVarsCLI
from seir.ode import CovidSeirODE
from dataclasses import asdict, fields
import numpy as np

from seir.ode import CovidSeirODE

args = DataClassArgumentParser([OdeParamCLI, LockdownCLI, MetaVarsCLI])
param_cli, lockdown_cli, meta_cli = args.parse_args_into_dataclasses()
ode = CovidSeirODE.from_cli(meta_cli, lockdown_cli, param_cli)


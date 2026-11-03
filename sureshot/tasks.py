"""The command line's steps as Python calls that return their results, for orchestrators: airflow.py, dagster.py and
prefect.py wrap run() so a pipeline can schedule calibrate, retrain, recheck, prune, stats, import, decide --batch...
and pass what each returns to the next task.

Options are the command's flags with underscores: run("calibrate", tier=1, sample=0.02) is `sureshot calibrate
--tier 1 --sample 0.02`, True is a bare flag, and a dict (such as costs) is sent as JSON."""
import json
from sureshot.cli import COMMANDS, parser

POSITIONAL = ("action", "file", "golden")  # label-studio push|pull, import FILE, test GOLDEN
LONG_RUNNING = ("serve", "mcp")


def run(command, **options):
    """What `sureshot <command>` prints, as data. A step that fails (no log, a bad option, a test below its
    --min-accuracy) raises RuntimeError with the command line's message."""
    if command in LONG_RUNNING or command not in COMMANDS:
        raise ValueError(f"no step {command!r}; the steps are {[c for c in COMMANDS if c not in LONG_RUNNING]}")
    argv = [command] + [str(options[key]) for key in POSITIONAL if options.get(key) is not None]
    for key, value in options.items():
        if key in POSITIONAL or value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        argv += [flag] if value is True else [flag, json.dumps(value) if isinstance(value, dict) else str(value)]
    try:
        return COMMANDS[command](parser().parse_args(argv))
    except SystemExit as e:
        message = e.code if isinstance(e.code, str) else f"sureshot {command} exited with status {e.code}"
        raise RuntimeError(message) from None

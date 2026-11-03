"""Dagster ops and assets for sureshot's steps (see tasks.py), whose output is what the step returns:

    from sureshot.dagster import sureshot_asset, sureshot_op
    thresholds = sureshot_asset("calibrate", tier=1, sample=0.02)
    report = sureshot_asset("stats", name="sureshot_stats", deps=[thresholds])

The code location needs sureshot installed and the same environment as the API (SURESHOT_DB, the model keys)."""
from dagster import In, Nothing, asset, op
from sureshot.tasks import run


def _name(command):
    return "sureshot_" + command.replace("-", "_")


def sureshot_op(command, name=None, **options):
    """In a job, start orders it after another op: recheck(start=calibrate())."""
    @op(name=name or _name(command), ins={"start": In(Nothing)})
    def step():
        return run(command, **options)
    return step


def sureshot_asset(command, name=None, deps=None, **options):
    @asset(name=name or _name(command), deps=deps)
    def step():
        return run(command, **options)
    return step

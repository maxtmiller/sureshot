"""A Prefect task for sureshot's steps (see tasks.py), returning what the step returns:

    from sureshot.prefect import sureshot
    report = sureshot("recheck", tier=1, sample=0.02)

The worker needs sureshot installed and the same environment as the API (SURESHOT_DB, the model keys)."""
from prefect import task
from sureshot.tasks import run


@task(task_run_name="sureshot {command}")
def sureshot(command, **options):
    return run(command, **options)

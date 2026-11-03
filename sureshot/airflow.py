"""An Airflow operator for sureshot's steps (see tasks.py). What the step returns is the task's XCom:

    from sureshot.airflow import SureshotOperator
    calibrate = SureshotOperator(task_id="calibrate", command="calibrate", options={"tier": 1, "sample": 0.02})

The worker needs sureshot installed and the same environment as the API (SURESHOT_DB, the model keys)."""
try:
    from airflow.sdk import BaseOperator  # Airflow 3
except ImportError:
    from airflow.models import BaseOperator
from sureshot.tasks import run


class SureshotOperator(BaseOperator):
    template_fields = ("options",)

    def __init__(self, *, command, options=None, **kwargs):
        super().__init__(**kwargs)
        self.command = command
        self.options = options or {}

    def execute(self, context):
        return run(self.command, **self.options)

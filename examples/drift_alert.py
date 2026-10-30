"""Exit with status 1 and print what moved when stats finds drift, so a scheduler can alert on it:

    */15 * * * * python examples/drift_alert.py hunch.db || <send the output wherever you get alerts>

The change detectors keep a fixed false alarm rate however often this runs."""
import json
import sys

from hunch import Store, stats

flagged = [{k: d[k] for k in ("tier", "key", "flags", "change", "upstream")} for d in stats(Store(sys.argv[1]))["drift"] if d["flags"]]
if flagged:
    print(json.dumps(flagged, indent=1))
    sys.exit(1)

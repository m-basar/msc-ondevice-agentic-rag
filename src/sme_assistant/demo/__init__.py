"""Post-evaluation dashboard demonstrator.

Built after the experiment was complete and frozen. Governed by
pre-registration amendment 1.27, which fixes the boundary this package must not
cross: nothing here contributes evidence to any hypothesis, and nothing here
writes into a frozen run directory.

Two modes, deliberately separated.

``Frozen Study Replay``
    Reads the committed records of the four frozen quality runs and shows all
    four arms side by side. Never invokes a model, and works on a machine with
    no Ollama installed.

``Live Assistant``
    Runs Arm D only, as the deployed system would, on a question the user
    types. Output is never scored and is written, if at all, under
    ``results/demo/``.

The four arms are never run live. A live four-arm comparison would look exactly
like the reported experiment and would not be it.

This package is imported by ``scripts/dashboard.py`` and by its tests, and by
nothing in ``sme_assistant.evaluation``. The dependency runs one way.
"""

from .replay import ReplayLibrary, ReplayUnavailable, load_replay_library

__all__ = ["ReplayLibrary", "ReplayUnavailable", "load_replay_library"]

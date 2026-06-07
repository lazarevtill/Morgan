"""Run brain-api: ``python -m morgan_brain.apps.brain_api``."""
from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("morgan_brain.apps.brain_api.app:app", host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()

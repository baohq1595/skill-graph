import copy
import json

import pytest

from skillgraph.context import build_context
from skillgraph.ingest.loader import DEFAULT_DATA_PATH


@pytest.fixture(scope="session")
def raw():
    return json.loads(DEFAULT_DATA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def ctx():
    return build_context()


@pytest.fixture
def write_dataset(tmp_path, raw):
    """Write a modified copy of the real dataset and return its path."""
    def _write(mutate):
        data = copy.deepcopy(raw)
        mutate(data)
        path = tmp_path / "dataset.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path
    return _write

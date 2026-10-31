"""Builds everything once: dataset -> validation -> graph -> resolver -> service."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from skillgraph.graph.store import NetworkXStore
from skillgraph.ingest.loader import DatasetError, data_path, load_dataset
from skillgraph.ingest.validate import ValidationReport, validate
from skillgraph.models import Dataset
from skillgraph.services.queries import SkillGraphService
from skillgraph.services.resolver import Resolver


@dataclass(frozen=True)
class AppContext:
    dataset: Dataset
    report: ValidationReport
    store: NetworkXStore
    resolver: Resolver
    service: SkillGraphService

    def learner(self, learner_id: str):
        for lp in self.dataset.learner_profiles:
            if lp.id.casefold() == learner_id.casefold():
                return lp
        raise KeyError(f"No sample learner {learner_id}. "
                       f"Available: {', '.join(lp.id for lp in self.dataset.learner_profiles)}")


def build_context(path: str | Path | None = None) -> AppContext:
    dataset = load_dataset(path)
    report = validate(dataset)
    if not report.ok:
        problems = "\n".join(f"- {i.code}: {i.message}" for i in report.errors)
        raise DatasetError(f"Ingestion stopped, the dataset has errors:\n{problems}")
    store = NetworkXStore(dataset)
    return AppContext(dataset, report, store, Resolver(store), SkillGraphService(store))


@lru_cache(maxsize=4)
def get_context(path: str | None = None) -> AppContext:
    return build_context(path or str(data_path()))

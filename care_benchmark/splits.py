"""Leave-one-turbine-out splits for CARE events."""
from __future__ import annotations

from typing import Dict, Iterator, List, Tuple

import pandas as pd

from care_benchmark.load_care import CareEvent


def leave_one_turbine_out_splits(
    events: List[CareEvent],
) -> Iterator[Tuple[int, List[CareEvent], List[CareEvent]]]:
    """Yield (held_out_asset_id, train_events, test_events) for each turbine."""
    asset_ids = sorted({event.asset_id for event in events})
    for asset_id in asset_ids:
        train = [e for e in events if e.asset_id != asset_id]
        test = [e for e in events if e.asset_id == asset_id]
        if train and test:
            yield asset_id, train, test


def group_events_by_asset(events: List[CareEvent]) -> Dict[int, List[CareEvent]]:
    grouped: Dict[int, List[CareEvent]] = {}
    for event in events:
        grouped.setdefault(event.asset_id, []).append(event)
    return grouped

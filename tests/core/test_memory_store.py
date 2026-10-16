"""Default memory time advances through eviction and persisted-record restoration."""

from worldfoundry.core.memory.store import MemoryQuery, MemoryRecord, MemoryStore


def test_eviction_preserves_append_time_and_recency_ranking() -> None:
    store = MemoryStore(capacity=2)
    for text in ("initial-a", "initial-b", "needle", "latest"):
        store.append(text, kind="text")
    assert [row["timestamp"] for row in store.records] == [2, 3]
    selection = store.select(MemoryQuery(text="needle", top_k=2, recency_weight=4))
    assert [(record.content, record.score) for record in selection.records] == [("latest", 4), ("needle", 3)]


def test_restored_and_explicit_timestamps_advance_default_time() -> None:
    store = MemoryStore(capacity=2, records=[MemoryRecord("old", timestamp=7), {"content": "new", "timestamp": 10}])
    assert store.append("next")["timestamp"] == 11
    store.append("explicit", timestamp=20.5)
    assert store.append("after-explicit")["timestamp"] == 21.5
    store.append("older-explicit", timestamp=1)
    assert store.append("after-older")["timestamp"] == 23.5


def test_reset_and_replace_start_a_new_memory_timeline() -> None:
    store = MemoryStore(capacity=2)
    store.append("old", timestamp=100)
    store.reset()
    assert store.append("fresh")["timestamp"] == 0
    store.replace([MemoryRecord("restored", timestamp=8)])
    assert store.append("next")["timestamp"] == 9
    store.replace(store.records)
    assert store.append("after-replacement")["timestamp"] == 10


def test_zero_capacity_still_advances_append_time() -> None:
    store = MemoryStore(capacity=0)
    assert [store.append(index)["timestamp"] for index in range(4)] == [0, 1, 2, 3]
    assert not store.records

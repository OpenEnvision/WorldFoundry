import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Example:
    x: Any
    features: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)


@dataclass
class Group:
    name: str | int = ""
    examples: list[Example] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    Ks: dict[str, Any] = field(default_factory=dict)
    ws: dict[str, Any] = field(default_factory=dict)
    vs: dict[str, Any] = field(default_factory=dict)


def to_batches(lst, batch_size):
    batches = []
    i = 0
    while i < len(lst):
        batches.append(lst[i : i + batch_size])
        i += batch_size
    return batches


def mode_dropping_groups(label_to_examples, categories, N):
    lst = []
    for i in range(1, len(categories) + 1):
        Ns = [N // i for _ in range(i)]
        while sum(Ns) < N:
            j = random.choice(range(len(Ns)))
            Ns[j] += 1
        group = Group(i, [])
        for n, cat in zip(Ns, categories, strict=False):
            group.examples += random.sample(label_to_examples[cat], n)
        lst.append(group)
    return lst

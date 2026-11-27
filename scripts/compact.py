"""The dataset as the site's build holds it: what evidence-core's `Dataset` answers, in a fraction of the memory.

evidence-core reads a dataset for tools that look at a few declarations: every edge becomes a Python integer in a
list, and a facet is parsed whole on first use and kept. A build of the whole library asks about every declaration
once or twice, and at 2a9233e5be32 (2026-11-27: 752,426 declarations, 642,550 of them the library's, 115 million
edges) holding the dataset that way took most of the build's 17.9 GB, which GitHub's 16 GB runner does not have.
Measured there, each piece alone in a process:

    the edges (meaning, statement, term)      4.6 GB    here 0.4 GB, as arrays (`Adjacency`)
    the facets signature and docs-signature   6.5 GB    here 0.05 GB: where each row is, read when asked (`Facet`)
    the facets source, axioms, roadmap,
    docstring                                 1.9 GB    likewise
    the declarations                          0.7 GB    here 0.5 GB, the strings they share held once

With the source files held one at a time (scripts/site_data.py, `OneSource`: 1.4 GB), the build peaks at 3.7 GB
there, and takes no longer: a row is parsed each time it is asked for, twice for most signatures, which costs
what the lists of integers did.

`CompactDataset` is a `Dataset` and answers every question the same: the build's output is the same, byte for byte
(the comparison is tests/test_compact.py's on the fixture, and was made by hand on the whole site at 2a9233e5be32).
Only two orders differ, and nothing reads either: `edges(notion).items()` goes by source id where evidence-core's
goes by first appearance in the file (the same order in every dataset so far, whose edge files are sorted), and
the lists of `reverse(notion)` are in the file's order, as evidence-core's are for such a file.
"""
from __future__ import annotations

import json
import operator
import os
import re
import sys
from array import array
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import accumulate, islice
from pathlib import Path

from evidence_core.dataset import SUPPORTED_SPECS, Dataset, Decl

#: The strings a declaration shares with many others: held once (`sys.intern`), where `json.loads` makes each anew.
SHARED = ("module", "package", "scope", "kind")
#: A row's declaration, found without parsing the row; `Facet` checks it against the row whenever it reads one.
DECL_KEY = re.compile(rb'"decl"\s*:\s*("(?:[^"\\]|\\.)*")')


class Adjacency(Mapping):
    """The adjacency lists of one notion (source id → target ids), as `Dataset.edges` gives them, in two arrays: the
    targets, by source, and where each source's start. A source's targets are an `array`, in the file's order; a
    declaration with none is not a key, as in evidence-core's dict."""
    __slots__ = ("_start", "_targets")

    def __init__(self, sources: array, targets: array, nodes: int):
        """From the pairs (sources[k], targets[k]), kept in their order under each source."""
        count = Counter(sources)
        nodes = max(nodes, max(count, default=-1) + 1)
        self._start = array("q", accumulate((count.get(i, 0) for i in range(nodes)), initial=0))
        if all(map(operator.le, sources, islice(sources, 1, None))):  # sorted by source, as trust-extract writes them
            self._targets = targets
        else:
            at, self._targets = array("q", self._start), array("i", bytes(4 * len(targets)))
            for s, t in zip(sources, targets):
                self._targets[at[s]] = t
                at[s] += 1

    def get(self, i, default=None):
        start = self._start
        try:
            lo, hi = start[i], start[i + 1]
        except (IndexError, TypeError):
            return default
        return self._targets[lo:hi] if lo < hi and i >= 0 else default

    def __getitem__(self, i):
        found = self.get(i)
        if found is None:
            raise KeyError(i)
        return found

    def __contains__(self, i) -> bool:
        return self.get(i) is not None

    def __iter__(self):
        start = self._start
        return (i for i in range(len(start) - 1) if start[i] < start[i + 1])

    def __len__(self) -> int:
        return sum(1 for _ in self)


class Facet(Mapping):
    """A facet (declaration name → its rows, usually one), as `Dataset.facet` gives it, read from its file a row at
    a time: this holds where each row is, and parses one when asked for it, so that going through the facet
    (`items()`) or through the declarations of a module holds a row at a time. Its names are in the file's order."""

    def __init__(self, path: Path, by_name: dict[str, Decl], nodes: int):
        self._by_name = by_name
        self._keys: list[str] = []          # the names, in the order they first appear
        self._where = array("q")            # each name's first row: where it starts, and how long it is
        self._length = array("q")
        self._at = array("i", [-1]) * nodes  # declaration id → its place in `_keys`
        self._others: dict[str, int] = {}   # likewise, for a name the dataset has no declaration of
        self._more: dict[int, list[tuple[int, int]]] = {}  # place → the name's rows after its first
        self._path, self._fd = path, os.open(path, os.O_RDONLY)
        offset = 0
        with open(path, "rb") as f:
            for line in f:
                if line.strip():
                    self._add(self._name(line), offset, len(line))
                offset += len(line)

    @staticmethod
    def _name(line: bytes) -> str:
        found = DECL_KEY.search(line) if line.count(b'"decl"') == 1 else None
        return json.loads(found.group(1)) if found else json.loads(line)["decl"]

    def _add(self, name: str, offset: int, length: int) -> None:
        k = self._place(name)
        if k is not None:
            self._more.setdefault(k, []).append((offset, length))
            return
        d = self._by_name.get(name)
        if d is not None and 0 <= d.id < len(self._at):
            self._at[d.id], name = len(self._keys), d.name  # the declaration's own string, not another copy
        else:
            self._others[name] = len(self._keys)
        self._keys.append(name)
        self._where.append(offset)
        self._length.append(length)

    def _place(self, name) -> int | None:
        if name in self._others:
            return self._others[name]
        d = self._by_name.get(name)
        if d is None or not 0 <= d.id < len(self._at) or self._at[d.id] < 0:
            return None
        return self._at[d.id]

    def _row(self, name: str, offset: int, length: int) -> dict:
        row = json.loads(os.pread(self._fd, length, offset))
        if row["decl"] != name:
            raise ValueError(f"{self._path}: the row at byte {offset} is of {row['decl']!r}, not of {name!r}")
        return row

    def __del__(self):
        fd, self._fd = getattr(self, "_fd", None), None
        if fd is not None:
            os.close(fd)

    def __getitem__(self, name) -> list[dict]:
        k = self._place(name) if isinstance(name, str) else None
        if k is None:
            raise KeyError(name)
        return [self._row(name, offset, length)
                for offset, length in [(self._where[k], self._length[k]), *self._more.get(k, ())]]

    def __contains__(self, name) -> bool:
        return isinstance(name, str) and self._place(name) is not None

    def __iter__(self):
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)


@dataclass
class CompactDataset(Dataset):
    """A dataset (S2) for a reader that goes through all of it: see the module's docstring."""

    _reversed: dict[str, Adjacency] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, root: str | Path) -> "CompactDataset":
        """As `Dataset.load`, with the strings the declarations share held once."""
        root = Path(root)
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        if meta.get("spec") not in SUPPORTED_SPECS:
            raise ValueError(f"{root}: unsupported dataset spec {meta.get('spec')!r}")
        decls = []
        with (root / "decls.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    for key in SHARED:
                        if isinstance(row.get(key), str):
                            row[key] = sys.intern(row[key])
                    decls.append(Decl.from_json(row))
        by_meaning: dict[str, list[Decl]] = {}
        for d in decls:
            if d.meaning:
                by_meaning.setdefault(d.meaning, []).append(d)
        return cls(root=root, meta=meta, decls=decls, by_name={d.name: d for d in decls}, by_meaning=by_meaning)

    def _pairs(self, notion: str) -> tuple[array, array]:
        """The edge file of a notion as (sources, targets), checked as `Dataset.edges` checks it."""
        entry = next((e for e in self.meta.get("edges", []) if e["name"] == notion), None)
        if entry is None:
            raise KeyError(f"no edge file for notion {notion!r} in {self.root}")
        if entry.get("format") != "i32le-pairs":
            raise ValueError(f"unsupported edge format {entry.get('format')!r}")
        data = array("i")
        data.frombytes((self.root / entry["file"]).read_bytes())
        if sys.byteorder != "little":
            data.byteswap()
        return data[0::2], data[1::2]

    def edges(self, notion: str) -> Adjacency:
        if notion not in self._edges:
            self._edges[notion] = Adjacency(*self._pairs(notion), len(self.decls))
        return self._edges[notion]

    def reverse(self, notion: str = "meaning") -> Adjacency:
        """Target id → source ids. Kept once made: evidence-core's `queue` asks for it at each call, and the build
        calls that twice."""
        if notion not in self._reversed:
            sources, targets = self._pairs(notion)
            self._reversed[notion] = Adjacency(targets, sources, len(self.decls))
        return self._reversed[notion]

    def facet(self, name: str) -> Mapping:
        if name not in self._facets:
            entry = next((f for f in self.meta.get("facets", []) if f["name"] == name), None)
            self._facets[name] = {} if entry is None else Facet(self.root / entry["file"], self.by_name, len(self.decls))
        return self._facets[name]

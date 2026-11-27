"""scripts/compact.py answers what evidence-core's `Dataset` answers, and the build holds the dataset through it: a
row, a file and a module at a time, where it once held all of them (17.9 GB at 2a9233e5be32, 2026-11-27, on a runner
with 16 GB). The tests of how much it holds are of its structure, on the fixture: no measurement, which would vary
from run to run."""
import json
import struct
import sys
import tempfile
import unittest
from array import array
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import build_site  # noqa: E402
import compact  # noqa: E402
import fixture_site  # noqa: E402
import site_data  # noqa: E402

from evidence_core.dataset import Dataset  # noqa: E402
from evidence_core.source import Sources  # noqa: E402

DATASET = fixture_site.DATASET
NOTIONS = ("meaning", "statement", "term", "source")


def lists(adjacency) -> dict[int, list[int]]:
    return {s: list(ts) for s, ts in adjacency.items()}


def files(root: Path) -> dict[str, bytes]:
    """Every file of a site by its path, meta.json without the time it was built at."""
    out = {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}
    meta = json.loads(out["data/meta.json"])
    del meta["built"]
    return {**out, "data/meta.json": meta}


class SameAnswersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plain, cls.compact = Dataset.load(DATASET), compact.CompactDataset.load(DATASET)

    def test_the_declarations(self):
        self.assertEqual(self.compact.decls, self.plain.decls)
        self.assertEqual(self.compact.by_name, self.plain.by_name)
        self.assertEqual(self.compact.by_meaning, self.plain.by_meaning)
        self.assertEqual(self.compact.meta, self.plain.meta)
        self.assertEqual(self.compact.modules, self.plain.modules)

    def test_the_strings_declarations_share_are_held_once(self):
        a, b = [d for d in self.compact.decls if d.is_project][:2]
        self.assertIs(a.scope, b.scope)
        self.assertIs(a.package, b.package)

    def test_the_edges(self):
        for notion in NOTIONS:
            want, got = self.plain.edges(notion), self.compact.edges(notion)
            self.assertEqual(lists(got), want, notion)
            self.assertEqual(len(got), len(want))
            for d in self.plain.decls:
                self.assertEqual(list(got.get(d.id, ())), want.get(d.id, []))
                self.assertEqual(d.id in got, d.id in want)
        self.assertIsNone(self.compact.edges("meaning").get(len(self.plain.decls) + 7))
        self.assertIsNone(self.compact.edges("meaning").get(-1))
        with self.assertRaises(KeyError):
            self.compact.edges("meaning")[-1]
        with self.assertRaises(KeyError):
            self.compact.edges("no such notion")

    def test_the_edges_reversed(self):
        for notion in NOTIONS:
            self.assertEqual(lists(self.compact.reverse(notion)), dict(sorted(self.plain.reverse(notion).items())))

    def test_what_follows_from_the_edges(self):
        for d in self.plain.decls:
            self.assertEqual(self.compact.successors(d.name), self.plain.successors(d.name))
            self.assertEqual(self.compact.closure(d.name), self.plain.closure(d.name))

    def test_the_facets(self):
        for name in [*self.plain.facet_names(), "no such facet"]:
            want, got = self.plain.facet(name), self.compact.facet(name)
            self.assertEqual(dict(got.items()), want, name)
            self.assertEqual(list(got), list(want), name)  # in the same order
            self.assertEqual(len(got), len(want))
            for d in self.plain.decls:
                self.assertEqual(self.compact.facet_row(name, d.name), self.plain.facet_row(name, d.name))
                self.assertEqual(d.name in got, d.name in want)
            self.assertIsNone(got.get("No.such.declaration"))
        for attr in ("claim", "specifies", "domain"):
            self.assertEqual(self.compact.annotations(attr), self.plain.annotations(attr))


class AdjacencyTests(unittest.TestCase):
    def test_pairs_in_no_order_keep_their_order_under_each_source(self):
        pairs = [(3, 9), (0, 5), (3, 1), (0, 4), (3, 9)]
        adj = compact.Adjacency(array("i", [s for s, _ in pairs]), array("i", [t for _, t in pairs]), nodes=2)
        self.assertEqual(lists(adj), {0: [5, 4], 3: [9, 1, 9]})
        self.assertEqual(list(adj), [0, 3])
        self.assertNotIn(1, adj)
        self.assertNotIn(4, adj)

    def test_no_pairs(self):
        self.assertEqual(lists(compact.Adjacency(array("i"), array("i"), nodes=3)), {})

    def test_it_holds_arrays_not_lists(self):
        adj = compact.CompactDataset.load(DATASET).edges("term")
        self.assertIsInstance(adj.get(next(iter(adj))), array)
        self.assertEqual(sum(len(ts) for ts in adj.values()),
                         len((DATASET / "edges" / "term.bin").read_bytes()) // struct.calcsize("ii"))


class FacetTests(unittest.TestCase):
    def facet(self, lines: list[str]) -> tuple[compact.Facet, dict]:
        """A facet of these lines, as `compact` reads it and as evidence-core does."""
        root = Path(tempfile.mkdtemp(prefix="compact-test-"))
        (root / "facets").mkdir()
        (root / "decls.jsonl").write_bytes((DATASET / "decls.jsonl").read_bytes())
        (root / "meta.json").write_text(json.dumps({"spec": "ltb-dataset/2",
                                                    "facets": [{"name": "odd", "file": "facets/odd.jsonl"}]}))
        (root / "facets" / "odd.jsonl").write_text("\n".join(lines), encoding="utf-8")
        return compact.CompactDataset.load(root).facet("odd"), Dataset.load(root).facet("odd")

    def test_rows_of_one_name_apart_names_the_dataset_lacks_and_blank_lines(self):
        got, want = self.facet([
            '{"decl":"Fixture.triple","n":1}', "", '{"n":2,"decl": "Not.in.the.dataset"}', "   ",
            '{"decl":"Fixture.double","text":"a \\"decl\\": \\"Fixture.triple\\" inside a text"}',
            '{"inner":{"decl":"Fixture.triple"},"decl":"Fixture.«term𝟚_»","n":4}',
            '{"decl":"Fixture.triple","n":5}', '{"decl":"Not.in.the.dataset","n":6}',
            '{"decl":"Fixture.\\u00abterm\\ud835\\udfda_\\u00bb","n":7}'])
        self.assertEqual(dict(got.items()), want)
        self.assertEqual(list(got), list(want))
        self.assertEqual(got["Fixture.triple"], [{"decl": "Fixture.triple", "n": 1}, {"decl": "Fixture.triple", "n": 5}])
        self.assertEqual(len(got["Fixture.«term𝟚_»"]), 2)
        self.assertNotIn("Fixture.IsDouble", got)
        with self.assertRaises(KeyError):
            got["Fixture.IsDouble"]

    def test_a_row_is_read_when_asked_for_and_not_kept(self):
        got, _ = self.facet(['{"decl":"Fixture.triple","n":1}', '{"decl":"Fixture.double","n":2}'])
        with mock.patch.object(compact.json, "loads", side_effect=json.loads) as loads:
            first, again = got["Fixture.double"], got["Fixture.double"]
        self.assertEqual(loads.call_count, 2)
        self.assertIsNot(first[0], again[0])


class BuildTests(unittest.TestCase):
    def test_the_site_is_the_same_file_for_file(self):
        """The build through `compact` and the build as it was, through evidence-core's own `Dataset` and
        `Sources`, write the same bytes (meta.json but for the time it was built at)."""
        root = Path(tempfile.mkdtemp(prefix="compact-test-"))
        fixture_site.build(root / "compact")
        with mock.patch.object(site_data, "CompactDataset", Dataset), mock.patch.object(site_data, "OneSource", Sources):
            fixture_site.build(root / "plain")
        got, want = files(root / "compact"), files(root / "plain")
        self.assertEqual(sorted(got), sorted(want))
        self.assertEqual([k for k in want if got[k] != want[k]], [])
        self.assertTrue(any(k.startswith("data/m/") for k in want))

    def test_the_build_holds_a_row_a_file_and_a_module_at_a_time(self):
        root = Path(tempfile.mkdtemp(prefix="compact-test-"))
        seen, held, steps = [], [], []
        site_of, module_file, write_packed = site_data.Site, site_data.Site.module_file, build_site.write_packed

        def site(inp, **kw):
            seen.append(site_of(inp, **kw))
            return seen[-1]

        def made(self, mi):
            steps.append(("made", mi))
            return module_file(self, mi)

        def written(path, data):
            steps.append(("written", int(path.name.split(".")[0])))
            held.append(len(seen[-1].inp.sources._files))
            return write_packed(path, data)

        with mock.patch.object(site_of, "module_file", made), mock.patch.object(site_data, "Site", site), \
                mock.patch.object(build_site, "write_packed", written):
            fixture_site.build(root / "site")
        ds = seen[-1].ds
        # Each module's file is written before the next is made: the build never holds two.
        modules = [mi for what, mi in steps if what == "made"]
        self.assertGreater(len(modules), 1)
        self.assertEqual(steps[:2 * len(modules)], [(what, mi) for mi in modules for what in ("made", "written")])
        # One source file at a time, though the modules' declarations are in several.
        self.assertGreater(len({r[0]["path"] for r in ds.facet("source").values()}), 1)
        self.assertEqual(max(held), 1)
        # The dataset's edges as arrays, and its facets as places in their files: no row, no list of integers.
        self.assertIsInstance(ds, compact.CompactDataset)
        self.assertTrue({"meaning", "statement", "term"} <= ds._edges.keys())
        self.assertTrue(all(isinstance(e, compact.Adjacency) for e in ds._edges.values()))
        self.assertTrue({"signature", "source", "docstring", "axioms", "roadmap"} <= ds._facets.keys())
        self.assertTrue(all(isinstance(f, compact.Facet) or f == {} for f in ds._facets.values()))


if __name__ == "__main__":
    unittest.main()

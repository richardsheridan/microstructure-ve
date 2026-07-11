"""The ABAQUS odb reader and its standalone-script generator, tested without ABAQUS.

``_read_abaqus_odb`` imports ``odbAccess`` / ``abaqusConstants`` at module top, so we
stub those in ``sys.modules`` before importing it and feed a hand-built fake odb object
graph. The generator's output is asserted to carry the reader source verbatim, so this
also exercises the script that ``write_odb_reader`` emits.
"""
import importlib
import io
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


class _Value:
    def __init__(self, data, conjugate=None):
        self.data = np.asarray(data, dtype=float)
        self.conjugateData = None if conjugate is None else np.asarray(conjugate, dtype=float)


class _Field:
    def __init__(self, values):
        self._values = values

    def getSubset(self, region=None, position=None):
        return self  # the fake already holds only the drive-node values

    @property
    def values(self):
        return self._values


class _Frame:
    def __init__(self, frame_value, u, rf):
        self.frameValue = frame_value
        self.fieldOutputs = {"U": u, "RF": rf}


class _Step:
    def __init__(self, frames, procedure="*STEADY STATE DYNAMICS, DIRECT"):
        self.frames = frames
        self.procedure = procedure


def _fake_odb():
    nset = object()
    instance = types.SimpleNamespace(nodeSets={"X1Y0": nset})
    root = types.SimpleNamespace(instances={"PART-1-1": instance})
    # one static (freq 0, skipped) frame + one dynamic frame with conjugate data
    static = _Frame(0.0, _Field([_Value([0.0, 0.0])]), _Field([_Value([0.0, 0.0])]))
    dynamic = _Frame(
        10.0,
        _Field([_Value([0.5, 0.0])]),
        _Field([_Value([2.0, -1.0], conjugate=[3.0, 0.5])]),
    )
    return types.SimpleNamespace(rootAssembly=root, steps={"Step-1": _Step([static, dynamic])})


class AbaqusOdbReaderTests(unittest.TestCase):
    def setUp(self):
        self.odb = _fake_odb()
        odbaccess = types.ModuleType("odbAccess")
        odbaccess.openOdb = lambda *a, **k: self.odb
        abqconst = types.ModuleType("abaqusConstants")
        abqconst.NODAL = "NODAL"
        self._patch = mock.patch.dict(
            sys.modules, {"odbAccess": odbaccess, "abaqusConstants": abqconst}
        )
        self._patch.start()
        mod_name = "microstructure_ve.backends.abaqus._read_abaqus_odb"
        sys.modules.pop(mod_name, None)  # force a fresh import under the stubs
        self.reader = importlib.import_module(mod_name)

    def tearDown(self):
        self._patch.stop()

    def test_read_odb_skips_zero_base_frame_and_sums_nodes(self):
        rows = self.reader.read_odb("job", "x1y0")
        self.assertEqual(len(rows), 1)  # the frame-value-0 base frame is skipped
        # [frame_value, RF_Real(2), RF_Imag(2), U_Real(2)]
        np.testing.assert_allclose(rows[0], [10.0, 2.0, -1.0, 3.0, 0.5, 0.5, 0.0])

    def test_static_step_keeps_only_the_final_frame(self):
        # An auto-incremented NLGEOM *STATIC step writes one frame per converged increment;
        # the oracle contract is one row per Static step (the FE side reports only the
        # converged step end), so the reader must keep the final loaded frame only.
        def static_frame(t, rf1):
            return _Frame(t, _Field([_Value([t * 0.5, 0.0])]),
                          _Field([_Value([rf1, 0.0])]))

        self.odb.steps = {"Step-1": _Step(
            [static_frame(0.0, 0.0), static_frame(0.25, 10.0),
             static_frame(0.5, 20.0), static_frame(1.0, 42.0)],
            procedure="*STATIC",
        )}
        rows = self.reader.read_odb("job", "x1y0")
        self.assertEqual(len(rows), 1)
        np.testing.assert_allclose(rows[0], [1.0, 42.0, 0.0, 0.0, 0.0, 0.5, 0.0])

    def test_no_dynamic_frames_raises(self):
        self.odb.steps = {"Step-1": _Step([_Frame(0.0, _Field([_Value([0.0, 0.0])]),
                                                  _Field([_Value([0.0, 0.0])]))])}
        with self.assertRaises(RuntimeError):
            self.reader.read_odb("job", "x1y0")

    def test_empty_drive_region_raises_naming_the_nodeset(self):
        # An empty RF/U subset (drive node set resolves to no values) must raise a clear
        # RuntimeError naming the node set, not an opaque IndexError on values[0].
        self.odb.steps = {"Step-1": _Step([_Frame(10.0, _Field([]), _Field([]))])}
        with self.assertRaises(RuntimeError) as cm:
            self.reader.read_odb("job", "x1y0")
        self.assertIn("x1y0", str(cm.exception))

    def test_write_tsv_header_and_rows(self):
        rows = [np.array([10.0, 2.0, -1.0, 3.0, 0.5, 0.5, 0.0])]
        with tempfile.TemporaryDirectory() as d:
            name = str(pathlib.Path(d) / "job")
            self.reader.write_tsv(name, rows)
            text = pathlib.Path(name + "-reaction-force.tsv").read_text()
        header = text.splitlines()[0].split("\t")
        self.assertEqual(
            header,
            ["frame_value", "RF_Real1", "RF_Real2", "RF_Imag1", "RF_Imag2", "U1", "U2"],
        )


class WriteOdbReaderTests(unittest.TestCase):
    def test_generated_script_carries_reader_source_verbatim(self):
        from microstructure_ve.backends.abaqus import write_odb_reader
        from microstructure_ve.backends.abaqus import _odb as odb

        with tempfile.TemporaryDirectory() as d:
            dest = pathlib.Path(d) / "read_abaqus_odb.py"
            out = write_odb_reader(dest)
            generated = out.read_text(encoding="utf-8")
        self.assertTrue(generated.startswith(odb.BANNER))
        self.assertTrue(generated.endswith(odb.reader_source()))
        # the emitted standalone script must not import the package
        self.assertNotIn("import microstructure_ve", generated)
        self.assertNotIn("from microstructure_ve", generated)


if __name__ == "__main__":
    unittest.main()

"""Public backend import surface.

Each backend exposes its user-facing API from a public module
(``microstructure_ve.backends.abaqus`` / ``...backends.dolfinx``); the implementation
lives in private ``_*`` submodules. Importing the dolfinx *package* (and its pure-numpy
``_spec``) must not require dolfinx, so the numpy-only tier can use it.
"""
import importlib
import unittest

from tests._helpers import HAS_DOLFINX


class AbaqusBackendApiTests(unittest.TestCase):
    def test_public_functions_importable(self):
        from microstructure_ve.backends.abaqus import write_inp, write_odb_reader

        self.assertTrue(callable(write_inp))
        self.assertTrue(callable(write_odb_reader))

    def test_implementation_modules_are_private(self):
        importlib.import_module("microstructure_ve.backends.abaqus._inp")
        importlib.import_module("microstructure_ve.backends.abaqus._odb")
        # the old non-underscore names should be gone
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("microstructure_ve.backends.abaqus.inp")


class DolfinxBackendApiTests(unittest.TestCase):
    def test_package_and_spec_import_without_dolfinx(self):
        # the dolfinx backend package init must stay light (no eager dolfinx import)
        importlib.import_module("microstructure_ve.backends.dolfinx")
        importlib.import_module("microstructure_ve.backends.dolfinx._spec")

    def test_implementation_modules_are_private(self):
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("microstructure_ve.backends.dolfinx.spec")

    @unittest.skipUnless(HAS_DOLFINX, "needs a DOLFINx env (dolfinx)")
    def test_run_is_public(self):
        from microstructure_ve.backends.dolfinx import run

        self.assertTrue(callable(run))

    @unittest.skipIf(HAS_DOLFINX, "only exercises the missing-dolfinx error path")
    def test_missing_dolfinx_gives_actionable_error(self):
        import microstructure_ve.backends.dolfinx as d

        with self.assertRaises(ImportError) as cm:
            d.run  # lazy import of the FE stack fails (basix/dolfinx absent)
        msg = str(cm.exception).lower()
        self.assertIn("dolfinx", msg)  # names the backend
        self.assertIn("basix", msg)    # names the actual missing dependency
        # box-agnostic: must not hardcode a specific env name or path
        self.assertNotIn("conda activate", msg)
        self.assertNotIn("docs/", msg)

    @unittest.skipUnless(HAS_DOLFINX, "needs a DOLFINx env (dolfinx)")
    def test_missing_threadpoolctl_gives_actionable_error(self):
        # threadpoolctl is part of the backend's FE stack: without it the entry points do not
        # resolve, with the same box-agnostic error a missing dolfinx gives. A child process
        # hides the module so the rest of the suite keeps it.
        import json
        import subprocess
        import sys

        child = (
            "import json, sys\n"
            "sys.modules['threadpoolctl'] = None\n"
            "import microstructure_ve.backends.dolfinx as d\n"
            "out = {}\n"
            "for name in ('run', 'build_solver'):\n"
            "    try:\n"
            "        getattr(d, name)\n"
            "        out[name] = None\n"
            "    except ImportError as e:\n"
            "        out[name] = str(e)\n"
            "print(json.dumps(out))\n"
        )
        proc = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True,
                              timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        results = json.loads(proc.stdout.strip().splitlines()[-1])
        for name, msg in results.items():
            with self.subTest(name):
                self.assertIsNotNone(msg, f"{name} resolved without threadpoolctl")
                msg = msg.lower()
                self.assertIn("threadpoolctl", msg)  # names the missing dependency
                self.assertIn("dolfinx", msg)        # names the backend
                self.assertNotIn("conda activate", msg)
                self.assertNotIn("docs/", msg)


if __name__ == "__main__":
    unittest.main()

"""Step subsections, Model construction-time validation, and Simulation assembly."""
import io
import unittest

import numpy as np

from microstructure_ve.backends.abaqus._inp import emit
from microstructure_ve.boundary import PeriodicBoundaryCondition
from microstructure_ve.core import ElementSet, GridElements, GridNodes
from microstructure_ve.materials import Material
from microstructure_ve.steps import (
    Dynamic,
    Heading,
    Model,
    Simulation,
    Static,
    Step,
)


def _emit(obj):
    buf = io.StringIO()
    emit(obj, buf)
    return buf.getvalue()


def _small_model_parts():
    img = np.zeros((3, 3), dtype=int)  # single material
    nodes = GridNodes.from_matl_img(img, 1.0)
    elements = GridElements(nodes, type="CPE4R")
    (elset,) = ElementSet.from_matl_img(img)
    materials = [Material(elset, density=1.0, poisson=0.3, youngs=1.0)]
    return nodes, elements, materials


class ModelValidationTests(unittest.TestCase):
    def test_overconstraint_raises_at_construction(self):
        nodes, elements, materials = _small_model_parts()
        with self.assertRaises(ValueError):
            Model(
                nodes=nodes,
                elements=elements,
                materials=materials,
                bcs=[
                    PeriodicBoundaryCondition(nodes=nodes),
                    PeriodicBoundaryCondition(nodes=nodes),
                ],
            )

    def test_valid_model_constructs(self):
        nodes, elements, materials = _small_model_parts()
        Model(nodes=nodes, elements=elements, materials=materials,
              bcs=[PeriodicBoundaryCondition(nodes=nodes)])


class SubsectionTextTests(unittest.TestCase):
    def test_heading(self):
        self.assertEqual(_emit(Heading("Hi")), "*Heading\nHi\n")

    def test_static(self):
        self.assertEqual(_emit(Static()), "*STATIC\n")
        self.assertEqual(_emit(Static(long_term=True)), "*STATIC, LONG TERM\n")

    def test_dynamic(self):
        text = _emit(Dynamic(f_initial=1e-7, f_final=1e5, f_count=30, bias=1))
        self.assertIn("*STEADY STATE DYNAMICS, DIRECT", text)
        self.assertIn("1e-07, 100000.0, 30, 1", text)

    def test_step_perturbation_flag(self):
        self.assertTrue(_emit(Step(subsections=[], perturbation=True)).startswith("*STEP,PERTURBATION"))
        self.assertTrue(_emit(Step(subsections=[])).startswith("*STEP\n"))
        self.assertIn("*END STEP", _emit(Step(subsections=[])))


class SimulationTests(unittest.TestCase):
    def test_assembles_heading_model_steps_in_order(self):
        nodes, elements, materials = _small_model_parts()
        model = Model(nodes=nodes, elements=elements, materials=materials,
                      bcs=[PeriodicBoundaryCondition(nodes=nodes)])
        sim = Simulation(heading=Heading("RVE"), model=model,
                         steps=[Step(subsections=[Static()])])
        text = _emit(sim)
        self.assertLess(text.index("*Heading"), text.index("*Node"))
        self.assertLess(text.index("*Node"), text.index("*STEP"))
        self.assertIn("*END STEP", text)


if __name__ == "__main__":
    unittest.main()

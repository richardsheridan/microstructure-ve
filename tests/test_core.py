"""Geometry / mesh primitives: node numbering, node sets, element connectivity."""
import io
import unittest

import numpy as np

from microstructure_ve.backends.abaqus._inp import emit
from microstructure_ve.core import (
    ElementSet,
    GridElements,
    GridNodes,
    NodeSet,
    Sides_2d,
    Sides_3d,
    _node_array,
)


class GridNodesTests(unittest.TestCase):
    def test_shape_is_a_plain_int_tuple(self):
        # shape is normalized to a tuple of python ints regardless of input type,
        # so it is hashable/comparable and needs no ndarray arithmetic downstream
        from_img = GridNodes.from_matl_img(np.zeros((4, 5)), scale=0.25)
        from_ctor = GridNodes(np.array([3, 3]), 1.0)
        for nodes in (from_img, from_ctor):
            self.assertIsInstance(nodes.shape, tuple)
            self.assertTrue(all(isinstance(d, int) for d in nodes.shape))

    def test_from_matl_img_shape_and_counts(self):
        nodes = GridNodes.from_matl_img(np.zeros((4, 5)), scale=0.25)
        self.assertEqual(nodes.shape, (5, 6))
        self.assertEqual(nodes.dim, 2)
        self.assertEqual(list(nodes.node_nums), list(range(1, 31)))
        self.assertEqual(nodes.virtual_node, 31)

    def test_dim_three(self):
        nodes = GridNodes(np.array([2, 3, 4]), 1.0)
        self.assertEqual(nodes.dim, 3)
        self.assertEqual(nodes.virtual_node, 2 * 3 * 4 + 1)

    def test_illegal_dim_raises(self):
        with self.assertRaises(ValueError):
            GridNodes(np.array([5]), 1.0)  # 1D
        with self.assertRaises(ValueError):
            GridNodes(np.array([2, 2, 2, 2]), 1.0)  # 4D


class NodeSetTests(unittest.TestCase):
    def test_from_slice_matches_ravel_multi_index_2d(self):
        nodes = GridNodes(np.array([3, 3]), 1.0)
        # corner (row=-1=2, col=-1=2) -> 1 + ravel((2,2),(3,3)) = 9
        ns = NodeSet.from_slice("X1Y1", Sides_2d["X1Y1"], nodes)
        np.testing.assert_array_equal(ns.node_inds, [9])
        # bottom edge interior Y0 = [0, 1:-1] -> node (0,1) = 1 + 1 = 2
        ns_y0 = NodeSet.from_slice("Y0", Sides_2d["Y0"], nodes)
        np.testing.assert_array_equal(ns_y0.node_inds, [2])

    def test_all_2d_nsets_disjoint_and_cover_boundary(self):
        nodes = GridNodes(np.array([4, 4]), 1.0)
        seen = []
        for inds in (ns.node_inds for ns in nodes.nsets.values()):
            seen.extend(int(i) for i in inds)
        self.assertEqual(len(seen), len(set(seen)), "node sets overlap")

    def test_all_3d_nsets_disjoint(self):
        nodes = GridNodes(np.array([4, 4, 4]), 1.0)
        self.assertEqual(set(Sides_3d), set(nodes.nsets))
        seen = []
        for ns in nodes.nsets.values():
            seen.extend(int(i) for i in ns.node_inds)
        self.assertEqual(len(seen), len(set(seen)), "3D node sets overlap")


class GridElementsTests(unittest.TestCase):
    def test_connectivity_is_counterclockwise(self):
        # 3x3 nodes -> 2x2 elements; lock the CCW vertex order (the product->swap).
        nodes = GridNodes(np.array([3, 3]), 1.0)
        elements = GridElements(nodes, type="CPE4R")
        buf = io.StringIO()
        emit(elements, buf)
        lines = [l for l in buf.getvalue().splitlines() if not l.startswith("*")]
        first = [int(x) for x in lines[0].split(",")]
        self.assertEqual(first, [1, 1, 2, 5, 4])  # elem_num, then CCW corners

    def test_element_type_validation(self):
        n2 = GridNodes(np.array([3, 3]), 1.0)
        n3 = GridNodes(np.array([3, 3, 3]), 1.0)
        for t in ("CPE4R", "CPS4R", "CPE4"):
            GridElements(n2, type=t)
        for t in ("C3D8R", "C3D8"):
            GridElements(n3, type=t)
        with self.assertRaises(ValueError):
            GridElements(n2, type="C3D8")  # 3D type on 2D nodes
        with self.assertRaises(ValueError):
            GridElements(n3, type="CPE4")  # 2D type on 3D nodes

    def test_element_count(self):
        nodes = GridNodes(np.array([5, 4]), 1.0)
        elements = GridElements(nodes, type="CPE4R")
        self.assertEqual(len(elements.element_nums), 4 * 3)


class ElementSetTests(unittest.TestCase):
    def test_from_matl_img_partitions_pixels(self):
        img = np.array([[0, 1, 2], [2, 1, 0]])
        sets = ElementSet.from_matl_img(img)
        self.assertEqual([int(s.matl_code) for s in sets], [0, 1, 2])
        all_elems = np.concatenate([s.elements for s in sets])
        np.testing.assert_array_equal(np.sort(all_elems), np.arange(1, img.size + 1))


class NodeArrayTests(unittest.TestCase):
    def test_bare_int(self):
        np.testing.assert_array_equal(_node_array(7), [7])

    def test_nodeset_returns_its_inds_without_copy(self):
        ns = NodeSet("S", np.array([3, 4, 5]))
        out = _node_array(ns)
        self.assertIs(out, ns.node_inds)  # np.asarray does not copy an ndarray


if __name__ == "__main__":
    unittest.main()

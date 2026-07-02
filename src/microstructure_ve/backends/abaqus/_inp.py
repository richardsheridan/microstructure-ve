"""ABAQUS ``.inp`` emission for the solver-neutral spec dataclasses.

Each spec object is serialized by a ``functools.singledispatch`` ``emit`` handler whose
body is the verbatim text the corresponding ``to_inp`` method used to write, so output
is byte-for-byte unchanged. Container objects (Model, Simulation, Step, the periodic
constraints, the materials) recurse through ``emit`` instead of calling child
``to_inp`` methods.
"""
from __future__ import annotations

from functools import singledispatch
from itertools import product

import numpy as np

from microstructure_ve.boundary import (
    DisplacementBoundaryCondition,
    FixedBoundaryCondition,
    OldPeriodicBoundaryCondition,
    PeriodicBoundaryCondition,
)
from microstructure_ve.core import ElementSet, GridElements, GridNodes, NodeSet
from microstructure_ve.equations import (
    DriveEquation,
    EqualityEquation,
    SequentialDifferenceEquation,
)
from microstructure_ve.constitutive import (
    Elastic,
    Plastic,
    PronyViscoelastic,
    TabularViscoelastic,
)
from microstructure_ve.materials import Material
from microstructure_ve.steps import (
    Dynamic,
    Heading,
    Model,
    Simulation,
    Static,
    Step,
)


@singledispatch
def emit(obj, f):
    """Write the ABAQUS ``.inp`` representation of ``obj`` to text file ``f``."""
    raise NotImplementedError(f"no ABAQUS emitter for {type(obj).__name__}")


def write_inp(simulation, file_or_path):
    """Emit ``simulation`` to a path or an already-open text file."""
    if hasattr(file_or_path, "write"):
        emit(simulation, file_or_path)
        return file_or_path
    with open(file_or_path, mode="w", encoding="ascii") as f:
        emit(simulation, f)
    return file_or_path


@emit.register(Heading)
def _(obj, f):
    f.write(
        f"""\
*Heading
{obj.text}
"""
    )


@emit.register(GridNodes)
def _(obj, f):
    # [::-1] reverses the [Z, Y, X] index axes into physical (x, y, z); see Sides_3d in core.
    coords = [np.ravel(c) for c in obj.scale * np.indices(obj.shape)[::-1]]
    f.write("*Node\n")
    for node_num, *p in zip(obj.node_nums, *coords):
        f.write(f"{node_num:d}")
        for d in p:
            f.write(f",\t{d:.6e}")
        f.write("\n")
    # place a "virtual" node co-located with the last (far corner) node
    f.write(f"{obj.virtual_node:d}")
    for c in coords:
        f.write(f",\t{c[-1]:.6e}")
    f.write("\n")
    for nset in obj.nsets.values():
        emit(nset, f)


@emit.register(GridElements)
def _(obj, f):
    # strategy: generate one array representing all nodes, then make slices of it
    # that represent offsets to e.g. the right or top to iterate
    all_nodes = 1 + np.ravel_multi_index(
        np.indices(obj.nodes.shape), obj.nodes.shape
    )
    node_slices = list(product(
        (np.s_[:-1], np.s_[1:]),
        repeat=obj.nodes.dim
    ))

    # elements are defined counterclockwise, but product produces zigzag
    # swapping the third and fourth elements works for squares and cubes
    node_slices[2], node_slices[3] = node_slices[3], node_slices[2]
    try:
        node_slices[6], node_slices[7] = node_slices[7], node_slices[6]
    except IndexError:
        pass  # it's 2D

    f.write(f"*Element, type={obj.type}\n")
    for elem_num, *ns in zip(
        obj.element_nums,
        *(all_nodes[sl].ravel() for sl in node_slices),
    ):
        f.write(f"{elem_num:d}")
        for n in ns:
            f.write(f",\t{n:d}")
        f.write("\n")


@emit.register(NodeSet)
def _(obj, f):
    f.write(f"*Nset, nset={obj.name}\n")
    for i in obj.node_inds:
        f.write(f"{i:d}\n")


@emit.register(ElementSet)
def _(obj, f):
    f.write(f"*Elset, elset=SET-{obj.matl_code:d}\n")
    for element in obj.elements:
        f.write(f"{element:d}\n")


@emit.register(SequentialDifferenceEquation)
def _(obj, f):
    for node0, node1 in zip(obj.nsets[0].node_inds, obj.nsets[1].node_inds):
        f.write(
            f"""\
*Equation
4
{node0}, {obj.dof}, 1.
{node1}, {obj.dof}, -1.
{obj.nsets[2]}, {obj.dof}, -1.
{obj.nsets[3]}, {obj.dof}, 1.
"""
        )


@emit.register(EqualityEquation)
def _(obj, f):
    f.write(
        f"""\
*Equation
2
{obj.nsets[0]}, {obj.dof}, 1.
{obj.nsets[1]}, {obj.dof}, -1.
"""
    )


@emit.register(DriveEquation)
def _(obj, f):
    f.write(
        f"""\
*Equation
3
{obj.nsets[0]}, {obj.dof}, 1.
{obj.nsets[1]}, {obj.dof}, -1.
{obj.drive_node}, {obj.dof}, 1.
"""
    )


def _emit_material_base(material, response, f, elastic_moduli=None):
    """The shared ``*Solid Section`` / ``*Material`` / ``*Elastic`` block.

    ``material`` supplies ``elset``/``density``; ``response`` supplies the elastic constants.
    ``elastic_moduli`` tags the ``*Elastic`` keyword (e.g. ``"LONG TERM"`` for a Prony response
    whose ``youngs``/``poisson`` are the relaxed moduli); ``None`` emits the plain ``*Elastic``
    used by every other response, byte-for-byte unchanged.
    """
    emit(material.elset, f)
    mc = material.elset.matl_code
    elastic = "*Elastic" if elastic_moduli is None else f"*Elastic, moduli={elastic_moduli}"
    f.write(
        f"""\
*Solid Section, elset=SET-{mc:d}, material=MAT-{mc:d}
1.
*Material, name=MAT-{mc:d}
*Density
{material.density:.6e}
{elastic}
{response.youngs:.6e}, {response.poisson:.6e}
"""
    )


@emit.register(Material)
def _(obj, f):
    # A Material is a container; dispatch the *Elastic / *Plastic / *Viscoelastic block on the
    # type of its constitutive response (the thing that actually varies), not the Material type.
    emit_response(obj.response, obj, f)


@singledispatch
def emit_response(response, material, f):
    """Write the ``*Elastic`` (+ ``*Plastic`` / ``*Viscoelastic``) block for ``response``."""
    raise NotImplementedError(f"no ABAQUS emitter for response {type(response).__name__}")


@emit_response.register(Elastic)
def _(response, material, f):
    _emit_material_base(material, response, f)


@emit_response.register(Plastic)
def _(response, material, f):
    _emit_material_base(material, response, f)
    f.write("*Plastic\n")
    for s, e in zip(response.yield_stress, response.plastic_strain):
        f.write(f"{s:.6e}, {e:.6e}\n")


@emit_response.register(TabularViscoelastic)
def _(response, material, f):
    _emit_material_base(material, response, f)
    f.write("*Viscoelastic, frequency=TABULAR\n")
    wgstar, wkstar = response.normalize_constant_nu_modulus()
    freq = response.apply_shift()
    for wgr, wgi, wkr, wki, fr in zip(
        wgstar.real, wgstar.imag, wkstar.real, wkstar.imag, freq
    ):
        f.write(f"{wgr:.6e}, {wgi:.6e}, {wkr:.6e}, {wki:.6e}, {fr:.6e}\n")


@emit_response.register(PronyViscoelastic)
def _(response, material, f):
    # youngs/poisson are the LONG-TERM (relaxed) moduli in this package (see Material), so tag
    # *Elastic accordingly; ABAQUS then derives the instantaneous moduli from the Prony ratios.
    _emit_material_base(material, response, f, elastic_moduli="LONG TERM")
    # ABAQUS PRONY ratios g_i = G_i/G_0, k_i = K_i/K_0 are relative to the INSTANTANEOUS moduli
    # G_0 = G_inf + sum(G_i), K_0 = K_inf + sum(K_i), with the relaxed G_inf = E/(2(1+nu)) and
    # K_inf = E/(3(1-2nu)). This matches PronyViscoelastic.complex_modulus exactly.
    g_inf = response.youngs / (2 * (1 + response.poisson))
    k_inf = response.youngs / (3 * (1 - 2 * response.poisson))
    g_ratios = response.shear_modulus_coefficients / (
        g_inf + np.sum(response.shear_modulus_coefficients)
    )
    k_ratios = response.bulk_modulus_coefficients / (
        k_inf + np.sum(response.bulk_modulus_coefficients)
    )

    f.write("*Viscoelastic, frequency=PRONY\n")
    for g, k, t in zip(g_ratios, k_ratios, response.relaxation_times):
        f.write(f"{g:.6e}, {k:.6e}, {t:.6e}\n")


@emit.register(FixedBoundaryCondition)
def _(obj, f):
    f.write(
        f"""\
*Boundary
"""
    )
    for dof in obj.dofs:
        f.write(
            f"""\
{obj.node}, {dof}, {dof}
"""
        )


def _emit_displacement_bc(obj, f):
    f.write(
        f"""\
*Boundary, type=displacement
{obj.nset}, {obj.first_dof}, {obj.last_dof}, {obj.displacement}
"""
    )


@emit.register(DisplacementBoundaryCondition)
def _(obj, f):
    _emit_displacement_bc(obj, f)


@emit.register(PeriodicBoundaryCondition)
def _(obj, f):
    for eq in obj.equations:
        emit(eq, f)


@emit.register(OldPeriodicBoundaryCondition)
def _(obj, f):
    for node_pair, eq_pair in zip(obj.node_pairs, obj.eq_pairs):
        emit(node_pair[0], f)
        emit(node_pair[1], f)
        emit(eq_pair[0], f)
        emit(eq_pair[1], f)
    _emit_displacement_bc(obj, f)


@emit.register(Static)
def _(obj, f):
    f.write(
        f"""\
*STATIC{", LONG TERM" if obj.long_term else ""}
"""
    )


@emit.register(Dynamic)
def _(obj, f):
    f.write(
        f"""\
*STEADY STATE DYNAMICS, DIRECT
{obj.f_initial}, {obj.f_final}, {obj.f_count}, {obj.bias}
"""
    )


@emit.register(Step)
def _(obj, f):
    f.write(
        f"""\
*STEP{",PERTURBATION" if obj.perturbation else ""}
"""
    )
    for n in obj.subsections:
        emit(n, f)
    f.write(
        f"""\
*END STEP
"""
    )


@emit.register(Model)
def _(obj, f):
    emit(obj.nodes, f)
    for nset in obj.nsets:
        emit(nset, f)
    emit(obj.elements, f)
    for m in obj.materials:
        emit(m, f)
    for bc in obj.bcs:
        emit(bc, f)


@emit.register(Simulation)
def _(obj, f):
    if obj.heading is not None:
        emit(obj.heading, f)
    emit(obj.model, f)
    for step in obj.steps:
        emit(step, f)

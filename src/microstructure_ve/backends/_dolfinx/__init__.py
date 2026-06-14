"""DOLFINx/FEniCSx FE backend (concern-separated submodules; see Step 4).

Placeholder until the test-driven rewrite lands. dolfinx/petsc4py/dolfinx_mpc are
imported lazily inside the submodules that need them, never at package import, so this
package stays importable under the numpy-only msve env.
"""

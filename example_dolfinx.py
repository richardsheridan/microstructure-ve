"""DOLFINx counterpart of example.py: solve the example RVE with FEniCSx.

Builds the *same* solver-neutral Simulation as the confined verification case and runs
it through the DOLFINx backend, writing example_dolfinx-reaction-force.tsv (readODB-style
columns) for direct comparison against an ABAQUS run via verify_pbc.compare.

Run in the fenicsx conda env:
    /home/rjs80/miniconda3/envs/fenicsx/bin/python example_dolfinx.py

The backend imposes a *confined* uniaxial-x macro strain (see dolfinx_backend.py), so the
matching ABAQUS oracle is the confined case (verify_pbc.py "new_confined" -> example_new),
not example.py's free-lateral loading.
"""
import numpy as np

import verify_pbc
import dolfinx_backend as db

# Reuse the exact confined RVE (mesh, materials, periodic BC, drive, 30-freq sweep)
sim, nodes = verify_pbc.build_sim("new_confined")

# Evaluate at the ABAQUS run's own frequencies when available, so the comparison is
# point-for-point; otherwise fall back to the Dynamic's log-spaced grid.
oracle = "example_new-reaction-force.tsv"
try:
    freqs = np.loadtxt(oracle, skiprows=1)[:, 0]
    freqs = np.sort(freqs)
    print(f"using {len(freqs)} frequencies from {oracle}")
except OSError:
    freqs = db.default_frequencies(sim)
    print(f"using {len(freqs)} default log-spaced frequencies")

out = db.run(sim, freqs=freqs, output_path="example_dolfinx-reaction-force.tsv")
print("wrote example_dolfinx-reaction-force.tsv")
print("storage RF_Real1 range: %.4g .. %.4g" % (out[:, 1].min(), out[:, 1].max()))

"""DOLFINx counterpart of example.py: solve the example RVE with FEniCSx.

Builds the *same* solver-neutral Simulation as example.py's free-lateral case and runs it
through the DOLFINx backend, writing example_dolfinx-reaction-force.tsv (readODB-style
columns) for direct comparison against an ABAQUS run via verify_pbc.compare.

Run in the fenicsx conda env:
    /home/rjs80/miniconda3/envs/fenicsx/bin/python example_dolfinx.py

Defaults: free-lateral loading (E_yy floats so σ̄_yy=0, matching example.py) with B-bar
(selective reduced integration) so the result matches ABAQUS CPE4 to <0.1%. Pass
lateral="confined" / bbar=False to db.run for the other variants.
"""
import numpy as np

import verify_pbc
import dolfinx_backend as db

# Reuse the exact free-lateral RVE (mesh, materials, periodic BC, drive, 30-freq sweep)
sim, nodes = verify_pbc.build_sim("new_free")

# Evaluate at the ABAQUS run's own frequencies when available (point-for-point compare);
# the CPE4 (full-integration) oracle matches B-bar, CPE4R is reduced-integration.
oracle = "example_free_cpe4-reaction-force.tsv"
try:
    freqs = np.sort(np.loadtxt(oracle, skiprows=1)[:, 0])
    print(f"using {len(freqs)} frequencies from {oracle}")
except OSError:
    freqs = db.default_frequencies(sim)
    print(f"using {len(freqs)} default log-spaced frequencies")

out = db.run(sim, freqs=freqs, lateral="free", bbar=True,
             output_path="example_dolfinx-reaction-force.tsv")
print("wrote example_dolfinx-reaction-force.tsv")
print("storage RF_Real1 range: %.4g .. %.4g" % (out[:, 1].min(), out[:, 1].max()))

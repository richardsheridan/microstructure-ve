# Mechanical-MNIST data excerpts

Source: https://github.com/elejeune11/Mechanical-MNIST (MIT license,
Copyright (c) 2019 Emma Lejeune). Fetched verbatim by ``tools/fetch_mechanical_mnist.py``;
see that script for the exact upstream paths.

- ``{train,test}_data_{0..9}.txt`` -- the benchmark's own 20 sample bitmaps
  (``generate_dataset/input_data/``), 28x28 integers 0..255. These are MNIST images
  0-9 of the respective train/test sets (the first ten test digits render as
  7 2 1 0 4 1 4 9 5 9).
- ``summary_psi_{train,test}_first10.txt`` -- rows 0-9 of the benchmark's
  ``data/summary_psi_{train,test}_all.txt`` (10 x 13 each): total strain energy change
  of the uniaxial-extension FEA at the 13 displacement steps
  d = 0, 0.001, 0.01, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 12, 14 (domain height 28, so the
  final step is 50% nominal strain), each row baseline-subtracted (column 0 is 0).

Pairing assumption: sample bitmap ``i`` corresponds to summary row ``i``. This cannot
be proven from the text files alone, but the parity run itself is the empirical check --
a wrong pairing produces O(1) disagreement, not the observed mesh-convergence-level
agreement (see ``tools/mechanical_mnist_parity.py``).

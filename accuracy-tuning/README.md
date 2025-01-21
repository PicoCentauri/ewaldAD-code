# Accuracy Tuning

This runs the accuracy tuning test for a given target accuracy for crystal structures
that were replicated 16 times in each direction.

Files:

- **[run.py](run.py)**: runs the tuning
- **[helpers.py](helpers.py)**: file containing code to generate the crystals used for
  the tuning
- **results_method_dtype.pkl**: [Pickle](https://docs.python.org/3/library/pickle.html)
  file containing a
  [dictionary](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) with
  the tuning results. One for each `method` (*Ewald*, *PME*, *P3M*) and `dtype` (*32bit*
  and *64bit*).

The tuning was performed with `torch-pme` version 0.2, `torch==2.4.1`, and CUDA 12.4.1,
on a H100.

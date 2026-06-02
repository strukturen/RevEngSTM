# Reverse Engineering Strut-and-tie Models for Assessing Reinforced Concrete Structures

This repository contains the code for the journal paper <b>"Reverse engineering strut-and-tie models for assessing reinforced concrete structures"</b> published in <i>Structural Concrete</i> and authored by Karin Yu and Walter Kaufmann, available under https://doi.org/10.1002/suco.70637. 

The user should be familiar with strut-and-tie models both for structural design and assessment. The responsibility of the results lies with the user.

The proposed method uses concepts from discrete layout optimisation, nonconvex nonlinear geometry optimisation and sparsification to reverse engineer strut-and-tie models given the reinforcement layout, geometry, material and proportional load cases to maximise the load-bearing capacity of in-plane loaded reinforced concrete structures. 

Please check out the journal paper to learn about the limitations of the proposed method. Note that the results after Stage 2 might exhibit some variability due to the nonconvex nature of the problem. As IPOPT is a local optimiser, different starting conditions and numerical sensitivities in the solver might result in different local optima. However, the results remain qualitatively consistent.

## Folder structure
- data: the saved ground structure and final strut-and-tie model after Stage 2 for each example
- results: the plots of the resulting ground structure, and strut-and-tie models after Stages 1 and 2 for each example
- src: the source files of the code, the two files `src/stm_methods.py` and `src/stm_trusssystem.py` are from the GitHub: https://github.com/strukturen/StrutandTieModelling (licensed under Apache 2.0 license), also authored by Karin Yu.
- example files of structural problems are included in the main directory
- run_reverseEngSTM.py: file to run the method
- requirements.txt: requirements file

## Getting started

For the optimisation of Stage 1, the MOSEK optimiser is used and requires a license. To apply the proposed method on the examples, run `python run_reverseEngSTM.py EXAMPLE_NAME EPS_DBSCAN ADD_ARGUMENTS`, where `EXAMPLE_NAME` is the name of the example Python script, `EPA_DBSCAN` the initial radius of the DBSCAN algorithm and `ADD_ARGUMENTS` any additional arguments of the example file. 

For instance, to run Case A of the dapped-end beam example use the following command `python run_reverseEngSTM.py Example_deb_A.py 0.3`. 

### Disclaimer
Some code (e.g. function `calcB`in `src/solver.py`) is taken from the paper "A Python script for adaptive layout optimization of truss structures" by L. He, M. Gilbert and X. Song published in [<i>Structural and Multidisciplinary Optimisation</i>](https://doi.org/10.1007/s00158-019-02226-6) in 2019.

## Citing
If you are using the code or find it useful, please consider citing the journal paper.
```
@article{yu_reverse_2026,
    author = {Yu, Karin and Kaufmann, Walter},
    title = {Reverse engineering strut-and-tie models for assessing reinforced concrete structures},
    journal = {Structural Concrete},
    year = {2026},
    keywords = {geometry optimization, layout optimization, reinforced concrete, structural assessment, structural optimization, strut-and-tie models},
    doi = {https://doi.org/10.1002/suco.70637},
    url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/suco.70637},
}
```

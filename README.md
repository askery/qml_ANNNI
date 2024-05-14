# qml_ANNNI
This repository hosts an implementation of a Quantum Variational Classifier specifically tailored for the Anisotropic Next-Nearest-Neighbor Ising (ANNNI) model. The ANNNI model is a cornerstone in the field of condensed matter physics, describing a 2D lattice with complex interactions beyond the nearest neighbors. The repository is related to the following paper:

> **Detecting quantum phase transitions in a frustrated spin chain with a quantum classifier algorithm**<br>
> André J. Ferreira-Martins (IIF/UFRN), Leandro Silva (IIF/UFRN + IFSC/USP), Alberto Palhares (IIF/UFRN + DFTE/UFRN), Rodrigo Pereira (IIF/UFRN + DFTE/UFRN), Diogo O. Soares-Pinto (IFSC/USP)
Rafael Chaves (IIF/UFRN + SST/UFRB) and Askery Canabarro (IFSC/USP + UFAL)<br>
> https://arxiv.org/abs/2309.15339
>
> **Abstract:** *The classification of phases and the detection of phase transitions are central and challenging tasks in diverse fields. Within physics, it relies on the identification of order parameters and the analysis of singularities in the free energy and its derivatives. Here, we propose an alternative framework to identify quantum phase transitions. Using the axial next-nearest neighbor Ising (ANNNI) model as a benchmark, we show how machine learning can detect three phases (ferromagnetic, paramagnetic, and a cluster of the antiphase with the floating phase). Employing supervised learning, we show that transfer learning becomes possible: a machine trained only with nearest-neighbor interactions can learn to identify a new type of phase occurring when next-nearest-neighbor interactions are introduced. We also compare the performance of common classical machine learning methods with a version of the quantum k-nearest neighbors (qKNN) algorithm.*

________________________________

This repository is organized as follows:

- File `q_neighbors.py`: library file with all functions and classes used in the project;
- File `qml_phase_transition.ipynb`: notebook file with reading and pre-processing of data, as well as execution of the algorithms and saving of outputs;
- Folder `data`: contains all input data. Please see the `README` file therein to understand what is in each file;
- Folder `results`: csv files with results. Each file contains an observation ID as well as the probability of the given observation belonging to class 1. Please see the notebook file to understand how each file is created.

________________________________

You can install these dependencies using pip and our requirements.txt file:

``` 
pip install -r requirements.txt
```

# License:
This project is licensed under the MIT License - see the LICENSE.md file for details.

# Acknowledgments:
We acknowledge the contributions of the open-source community and related research in the field of quantum machine learning and condensed matter physics, which have been instrumental in the development of this repository.

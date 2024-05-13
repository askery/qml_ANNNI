# qml_ANNNI
This repository hosts an implementation of a Quantum Variational Classifier specifically tailored for the Anisotropic Next-Nearest-Neighbor Ising (ANNNI) model. The ANNNI model is a cornerstone in the field of condensed matter physics, describing a 2D lattice with complex interactions beyond the nearest neighbors. The repository is related to the following paper:

André J. Ferreira-Martins ,1 Leandro Silva,1,2 Alberto Palhares ,1,3 Rodrigo Pereira ,1,3 Diogo O. Soares-Pinto,2
Rafael Chaves,1,4 and Askery Canabarro 2,5,*
3
4
1International Institute of 5 Physics, Federal University of Rio Grande do Norte, 59078-970 Natal, Brazil
2Instituto de Física de São Carlos, Universidade de São Paulo, CP 369, 13560-970 São Carlos, São Paulo, Brazil 6
3Departamento de Física Teórica e Experimental, Federal University of Rio Grande do Norte, 59078-970 Natal, Brazil 7
4School of Science and Technology, Federal University of Rio Grande do Norte, 59078-970 Natal, Brazil 8
5Grupo de Física da Matéria Condensada, Núcleo de Ciências Exatas - NCEx, Campus Arapiraca, Universidade Federal de Alagoas,
57309-005 Arapiraca, Alagoas, Brazil

> **Detecting quantum phase transitions in a frustrated spin chain with a quantum classifier algorithm**<br>
> André J. Ferreira-Martins (IIF/UFRN), Leandro Silva (IIF/UFRN + IFSC/USP), Alberto Palhares (IIF/UFRN + DFTE/UFRN), Rodrigo Pereira (IIF/UFRN + DFTE/UFRN), Diogo O. Soares-Pinto (IFSC/USP)
Rafael Chaves (IIF/UFRN + SST/UFRB) and Askery Canabarro (IFSC/USP + UFAL)<br>
> https://arxiv.org/abs/XXXX.YYYYY
>
> **Abstract:** *The classification of phases and the detection of phase transitions are central and challenging tasks in diverse fields. Within physics, it relies on the identification of order parameters and the analysis of singularities in the free energy and its derivatives. Here, we propose an alternative framework to identify quantum phase transitions. Using the axial next-nearest neighbor Ising (ANNNI) model as a benchmark, we show how machine learning can detect three phases (ferromagnetic, paramagnetic, and a cluster of the antiphase with the floating phase). Employing supervised learning, we show that transfer learning becomes possible: a machine trained only with nearest-neighbor interactions can learn to identify a new type of phase occurring when next-nearest-neighbor interactions are introduced. We also compare the performance of common classical machine learning methods with a version of the quantum k-nearest neighbors (qKNN) algorithm.*

# Key Features:
**Anisotropic Next-Nearest-Neighbor Ising Model**: Implementations and utilities for simulating and handling the AnNNI model, enabling researchers to explore its behavior.

Quantum Variational Classifier: A robust and efficient quantum variational classifier designed to handle the specific challenges posed by the AnNNI model.

Optimization Strategies: Various optimization techniques are tailored for training the variational classifier on the AnNNI model, ensuring convergence and efficiency.

Benchmarking and Evaluation: Tools for benchmarking the performance of the classifier against classical machine learning algorithms on a range of AnNNI instances.

Documentation and Tutorials: Comprehensive documentation and tutorials to guide users in understanding, extending, and utilizing the codebase effectively.

# How to Use:
Clone the repository and follow the detailed documentation to set up the environment and get started.

Utilize the provided Jupyter notebooks for hands-on demonstrations and experiments.

Experiment with different hyperparameters, optimization strategies, and input configurations to fine-tune the classifier for specific AnNNI instances.

# Main Dependencies (tested only for the versions below)
Make sure you have the following dependencies installed:

qiskit==0.31.0

qiskit-aer==0.9.1

qiskit-aqua==0.9.5

qiskit-ibmq-provider==0.17.0

You can install these dependencies using pip and our requirements.txt file:

``` 
pip install -r requirements.txt
```

# License:
This project is licensed under the MIT License - see the LICENSE.md file for details.

# Acknowledgments:
We acknowledge the contributions of the open-source community and related research in the field of quantum machine learning and condensed matter physics, which have been instrumental in the development of this repository.

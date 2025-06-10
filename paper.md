---
title: 'QuantumReservoirPy: A Software Package for Time Series Prediction'
tags:
  - Python
  - physics
  - quantum information
  - quantum circuits
  - qiskit
authors:
  - name: Ola Tangen Kulseng
    orcid: 0009-0009-9807-4975
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Stanley Miao
    equal-contrib: true
    affiliation: 2 # (Multiple affiliations must be quoted)
  - name: Franz G. Fuchs
    orcid: 0000-0003-3558-503X
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 3
  - name: Alexander Stasik
    orcid: 0000-0003-1646-2472
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3, 4
affiliations:
 - name: David R. Cheriton School of Computer Science, University of Waterloo, Canada
   index: 2
 - name: Department of Physics, Norwegian University of Science and Technology (NTNU), Norway
   index: 1
 - name: Department of Mathematics and Cybernetics, SINTEF Digital, Norway
   index: 3
 - name: Department of Data Science, Norwegian University of Life Science, Norway
   index: 4
date: 22 February 2025
bibliography: paper.bib
---

# Summary
Reservoir computing on quantum computers has recently emerged as a potential resource for time series prediction, owing to its inherent complex dynamics. To advance Quantum Reservoir Computing (QRC) research, we have developed the Python package [`QuantumReservoirPy`](https://github.com/OpenQuantumComputing/quantumreservoirpy), which facilitates QRC using quantum circuits as reservoirs.

The package is designed to be easy to use, while staying completely customizable. The resulting interface, similar to that of [reservoirpy](https://github.com/reservoirpy/reservoirpy) [@trouvain20], simplifies the development of quantum reservoirs, and provides logical methods of comparison between reservoir architectures.


# Statement of need
Reservoir computing (RC) is a paradigm in machine learning for time series prediction. With recent developments, it has shown a promising efficacy compared to conventional neural networks, owing to its relatively simple training process [@tanaka2019].

The main building block of RC is a dynamical system called a reservoir. By making the dynamics of the reservoir dependent on the input time series, the state of the reservoir becomes a high-dimensional, non-linear transformation of the time series. The hope is that such a low-to-high dimensional encoding enables forecasting using a relatively simple method like Ridge regression. A reservoir can be virtual, such as a sparsely-connected recurrent neural network with random fixed weights, termed an echo state network [@jaeger04], or even physical, such as a bucket of water [@fernando03]. See \autoref{fig1} for an illustration of a typical RC pipeline.


![A quantum reservoir system consists of a learning task, an en- and de-coder (red), and the dynamic system itself (green). In standard RC the machine learning part is linear regression.\label{fig1}](fig1.pdf)


QRC is a proposed method of RC utilizing quantum circuits as reservoirs. Multi-qubit systems with the capability of quantum entanglement provide compelling non-linear dynamics that match the requirements for a feasible reservoir. Furthermore, the exponentially-scaling Hilbert space of large multi-qubit systems support the efficiency and state-storage goals of RC. As a result, quantum computers have been touted as a viable dynamical system to produce the intended effects of RC.

Existing implementations of QRC have used proprietary realizations on simulated and actual quantum computers. The lack of a shared structure between implementations has resulted in a disconnect in comparing reservoir architectures. In addition, individual implementations require a certain amount of redundant procedure prior to the involvement of specific concepts.

We observe that there is a need for a common framework for the implementation of QRC. As such, we have developed `QuantumReservoirPy` to solve the issues presented in current QRC research. `QuantumReservoirPy` is designed to handle every step of the RC pipeline, in addition to the pre- and post-processing needed in the quantum case. In providing this software package, we hope to facilitate logical methods of comparison in QRC architecture and enable a simplified process of creating a custom reservoir from off-the-shelf libraries with minimal overhead requirements to begin development.

![Quantum circuit construction may be customized through the `before`, `during`, and `after` methods and a timeseries processed with the `run` and `predict` methods.\label{fig2}](fig2.pdf)


# Design and implementation


We intend `QuantumReservoirPy` to provide flexibility to all possible designs of quantum reservoirs, with full control
over pre-processing, input, quantum circuit operations,
measurement, and post-processing. In particular, we take inspiration from the simple and flexible structure provided by the ReservoirPy software package [reservoirpy](https://github.com/reservoirpy/reservoirpy) [@trouvain20].


The construction methods of `QuantumReservoirPy` serve as sequential operations performed on the quantum system. All of them may include an arbitrary combination of operations from the [Qiskit Circuit Library](https://docs.quantum.ibm.com/api/qiskit/circuit_library). `before` and `after` are independent of the time series, and are applied before and after the time series is processed, respectively. Operations in `during` are applied per timestep, and most closely determine the dynamical properties of the reservoir. \autoref{fig3}a demonstrates the aforementioned arrangement, which is implemented as a hidden intermediary process in a `QuantumReservoirPy` quantum reservoir.

![The intended functionality of the `run` and `predict` method. The observed input sequence is $\{x_t\}$ and the target sequence $\{y_t\}$. The reservoir $f$ performs an evolution in time.\label{fig3}](fig3.pdf)

The processing methods do not affect the creation of the reservoirs, but are included to keep a coherent interface to `reservoirpy`. Calling `run` on a time series returns the non-linear embeddings for each timestep. Depending on the realization of QRC, such as averaging over multi-shot data, additional post-processing can be included in the `run` method to achieve the desired output. \autoref{fig3}b provides a visualization of the `run` method. `predict` functions as a complete forecasting process including encoding, decoding, and post-processing. Additionally, a trained simple machine learning model is used to predict the next step in the timeseries from the transformed and post-processed data. The resulting prediction is then fed in as input for the following prediction, which occurs as an iterative process until the specified number of forecasting steps is reached. At this point, the `predict` method returns the sequence of predictions from each iteration. \autoref{fig3}c provides a visualization of the `predict` method.

# Package Details

## Dependencies

The three main dependencies of `QuantumReservoirPy` are numpy, qiskit, and scikit-learn, with python versions above 3.9. Qiskit is [deprecating](https://github.com/Qiskit/qiskit/releases) python 3.9 support in the 2.1.0 version, and the package presented here is developed to support qiskit=2.0.x. As for the other packages, the supported versions of scikit-learn and numpy follows from their interrelated constraints as well as the constraint from qiskit. In the install script, we specify numpy>1.17. We strive for `QuantumReservoirPy` to support compatibility with existing reservoir computing and quantum computing workflows.

Much of existing research in QRC is performed on IBM devices and simulators (see @yasuda23 and @suzuki22), programmed through the Qiskit software package. To minimize disruption in current workflows, `QuantumReservoirPy` is built as a package to interact with Qiskit circuits and backends. It is expected that the user also uses Qiskit in the customization of reservoir architecture when working with `QuantumReservoirPy`.

## License
`QuantumReservoirPy` is licensed under the GNU General Public License v3.0. `QuantumReservoirPy` also includes derivative work of Qiskit, which is licensed by IBM under the Apache License, Version 2.0.

## Further Development
The authors continue to support and maintain the project. Users may report package issues and desired features by opening an issue on the public GitHub repository or contacting the authors by email. Additional opportunities for further development on `QuantumReservoirPy` include supplementary built-in processing schemes, expanded features for data visualization, and reservoir evaluation methods.


# Acknowledgements
Work in this project was supported by the NTNU and SINTEF Digital through the International Work-Integrated-Learning in Artificial Intelligence (IWIL AI) program, in partnership with SFI NorwAI and the Waterloo Artificial Intelligence Institute (Waterloo.AI). IWIL AI is funded by the Norwegian Directorate for Higher Education and Skills (HK-dir).


# References

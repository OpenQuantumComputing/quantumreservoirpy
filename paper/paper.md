---
title: 'QuantumReservoirPy: A Software Package for Time Series Prediction'
tags:
  - Python
  - astronomy
  - dynamics
  - galactic dynamics
  - milky way
authors:
  - name: Adrian M. Price-Whelan
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
  - given-names: Ludwig
    dropping-particle: van
    surname: Beethoven
    affiliation: 3
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University, United States
   index: 1
   ror: 00hx57361
 - name: Institution Name, Country
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 13 August 2017
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Abstract

In recent times, quantum reservoir computing has emerged as a potential resource for time series prediction. Hence, there is a need for a flexible framework to test quantum circuits as nonlinear dynamical systems. We have developed a software package to allow for quantum reservoirs to fit a common structure, similar to that of reservoirpy [@trouvain20], which is advertised as
 "a python tool designed to easily define, train and use [classical] reservoir computing architectures".
Our package results in simplified development and logical methods of comparison between quantum reservoir architectures. Examples are provided to demonstrate the resulting simplicity of executing quantum reservoir computing using our software package.

# Introduction
Reservoir computing (RC) is a paradigm in machine learning for time series prediction. With recent developments, it has shown a promising advantage in efficiency due to the relative simplicity of the associated training process over conventional neural network methods [@tanaka2019].
In reservoir computing, a dynamical system composed of hidden functional representations with non-linear state transitions is chosen as a reservoir. Input data from a time series is sequentially encoded and fed into the reservoir. The hidden parameters of the reservoir undergo a non-linear evolution dependent on the stored state and the encoded information fed into the system. Features are subsequently decoded from a readout of certain parameters of the reservoir, which is used to train a simple linear model for the desired time series prediction output, see Figure  ![Caption for example figure.\label{fig:example}](fig1.pdf) for an illustration.
The selected reservoir must incorporate non-linear state transitions within the limitations of a fixed structure. A reservoir can be virtual, such as a sparsely-connected recurrent neural network with random fixed weights, termed as an echo state network [@jaeger04] or even physical, such as a bucket of water [@fernando03].

RC can be phrased as a supervised machine learning problem. Typically, one has an observed input sequence $\{x_t\}$ and a target sequence $\{y_t\}$. The reservoir $f$ then performs an evolution in time given by $ u_{t+1} = f(u_t, x_t)$
where $u_t$ is the internal state of the reservoir at time $t$. The output of the reservoir is given by
$
\hat{y}_t = W_\text{out} h(u_t)
$
where $h$ is the observation function. If the reservoir is fully observed, $h$ reduces to the identity. $W_\text{out}$ is a matrix mapping the observation on the target sequence, typically by minimizing the loss $\mathcal{L} = \sum_{t} \|(\hat{y}_t - y_t)\|_p$. For $p=2$ this reduces to linear regression with mean squared error. The performance of the reservoir is determined by the dynamical properties of the reservoir $f$.

Quantum reservoir computing (QRC) is a proposed method of reservoir computing. Multi-qubit systems with the capability of quantum entanglement provide compelling non-linear dynamics that match the requirements for a feasible reservoir. Furthermore, the exponentially-scaling Hilbert space of large multi-qubit systems support the efficiency and state-storage goals of reservoir computing. As a result, quantum computers have been touted as a viable dynamical system to produce the intended effects of reservoir computing.

In QRC, data is encoded by operating on one or more qubit(s) to reach a desired state. To obtain the desired complex non-linearity in a quantum system as a reservoir, entangling unitary operations are performed over the system. The readout is measured from one or more qubit(s) of the quantum state, which can be achieved through partial or full measurement over the system. Since quantum measurement results in a collapse to the measured state, repetitive measurements over identical systems are used to sample from the distribution of the unknown quantum state, which is then post-processed to obtain a decoded measurement for the subsequent linear model. Furthermore, this collapse results in a loss of retained information and entanglement in the system. Where only a partial measurement is taken (as in [@yasuda23]), this may have a desired effect of a slow leak of information driven from earlier input. When measurement is taken over the full system, the system may instead be restored through re-preparation of the pre-existing system, achieved in [@suzuki22] and through the restarting and rewinding protocols in [@mujal23].

Existing implementations of QRC have used proprietary realizations on simulated and actual quantum computers. The lack of a shared structure between implementations has resulted in a disconnect with comparing reservoir architectures. In addition, individual implementations require a certain amount of redundant procedure prior to the involvement of specific concepts.
We observe that there is a need for a common framework for the implementation of QRC. As such, we have developed a software package,`QuantumReservoirPy`, to solve the presented issues in current QRC research. In providing this software package, we hope to facilitate logical methods of comparison in QRC architecture and enable a simplified process of creating a custom reservoir from off-the-shelf libraries with minimal overhead requirements to begin development.
# Software Package
We intend `QuantumReservoirPy` to provide flexibility to all possible designs of quantum reservoirs, with full control
over pre-processing, input, quantum circuit operations,
measurement, and post-processing. In particular, we take inspiration from the simple and flexible structure provided by the ReservoirPy software package [@trouvain20].
## Structure and Design
Unlike the parameterized single-class structure of ReservoirPy, `QuantumReservoirPy` uses an abstract class-based structure (see Figure~\ref{fig:qreservoir-structure}) as the former would be restrictive in the full customization of QRC. In particular, quantum circuit operations and measurement may be conducted in differing arrangements. With an abstract class, we allow the user to define the functionality of the quantum circuit, which provides full flexibility over all existing implementations of QRC. This structure also implicitly provides access to the full Qiskit circuit library.

The construction methods in `QuantumReservoirPy` serve as the sequential operations performed on the quantum system. The operations in the `before` method prepares the qubits, which may include initialization to a default, initial, or previously saved state. The `during` method provides the operations that are subsequently applied for each step in the timeseries. This may include (but is not limited to) measurement, re-initialization, and entangling operations. Finally, the operations in the `predict` method are applied once following the processing of the entire timeseries, which may include the transformation of final qubit states and measurement. Figure~\ref{fig:circuit-function} demonstrates the aforementioned arrangement, which is implemented as a hidden intermediary process in a `QuantumReservoirPy` quantum reservoir.

The processing methods serve as functions acting on the quantum reservoir itself. The `run` method is used to process training data by taking a timeseries as input and returning the transformed data after being processed by the quantum reservoir. This is done using the hidden `circuit` interface as presented in Figure~\ref{fig:circuit-function}, where data encoding and decoding follow the implementation of the custom construction methods. Depending on the realization of QRC, such as averaging over multi-shot data, additional post-processing is included in the `run` method to achieve the desired output. The transformed data from the `run` method serves as training data for a simple machine learning model. Figure~\ref{fig:run-function} provides a visualization of the `run` method.

The `predict` method functions as a complete forecasting process involving the same hidden `circuit` interface, encoding, decoding, and post-processing. Additionally, a trained simple machine learning model is used to predict the next step in the timeseries from the transformed and post-processed data. The resulting prediction is then fed in as input for the following prediction, which occurs as an iterative process until the specified number of forecasting steps is reached. At this point, the `predict` method returns the sequence of predictions from each iteration. Figure~\ref{fig:predict-function} provides a visualization of the `predict` method.

## Dependencies

The three main dependencies of `QuantumReservoirPy` are numpy, qiskit, and scikit-learn.
We strive for `QuantumReservoirPy` to support compatibility with existing reservoir computing and quantum computing workflows.

`QuantumReservoirPy` uses NumPy as a core dependency. Allowed versions of NumPy ensure compatibility with classical ReservoirPy to facilitate comparison between classical and quantum reservoir architectures. An example is provided [here](examples\static).

Much of existing research in QRC is performed on IBM devices and simulators (see [@yasuda23; @suzuki22]), programmed through the Qiskit software package. To minimize disruption in current workflows, `QuantumReservoirPy` is built as a package to interact with Qiskit circuits and backends. It is expected that the user also use Qiskit in the customization of reservoir architecture when working with `QuantumReservoirPy`.

The model used by the `predict` function is expected to be a scikit-learn estimator. Likewise to our dependency on Qiskit, this allows for complete customization over the choice of model and faster adoption of `QuantumReservoirPy` from a simple package structure.

## Distribution
The software package is developed and maintained through a public GitHub repository provided through the `OpenQuantumComputing` organization. The `main` branch of this repository serves as the latest stable version of `QuantumReservoirPy`. Installation through cloning this repository is intended for development purposes.

Official releases of the software package are published through the Python Package Index (PyPI) under the `quantumreservoirpy` name. Installation through PyPI is the suggested method of installation for general-purpose use.

`QuantumReservoirPy` is licensed under the GNU General Public License v3.0. `QuantumReservoirPy` also includes derivative work of Qiskit, which is licensed by IBM under the Apache License, Version 2.0.
## Documentation
Documentation is mainly provided online in the form of a user guide and API reference. The user guide includes steps for installation and getting started with a simple quantum reservoir. Examples are also provided for further guidance. The API reference outlines the intended classes and functions exposed by the software package.
A brief overview of the software package is also included as a `README.md` file at the top level of the public GitHub repository.

# Usage

`QuantumReservoirPy` provides two prepackaged reservoirs that implement the processing methods `run` and `predict` according to the structure presented in section \ref{sec:structure-design}. These reservoirs use common QRC schemes as a basis to get started with the software package or restrict customization to circuit construction. As such, the construction of a reservoir using either of these schemes only requires user implementation of the sequential circuit methods `before`, `during`, and `after`.

## Static Reservoirs
The `Static` abstract reservoir class is structured to run multi-shot processing according to a repeated-measurement process. A single circuit is created according to the circuit construction methods and the timeseries parameter. Measurement is expected in the `during` function to generate the transformed reservoir output using the single circuit. This circuit is sent to the specified Qiskit backend and is run with the remaining parameters provided to the processing methods. The resulting measurement data is post-processed by taking the average over all shots before it is returned to the user as decoded reservoir output.

An example of a `Static` quantum reservoir is the following.

```python
class QuantumReservoir(Static):
    def before(self, circuit):
        circuit.h(circuit.qubits)

    def during(self, circuit, timestep):
        circuit.initialize(encoder[timestep], [0, 1])
        circuit.append(operator, circuit.qubits)
        circuit.measure([0, 1])

```

In the example, the timestep passed to the `during` method is encoded in the circuit according to an `encoder`. Measurement is also taken in the `during` method to provide sequential data using a single circuit for the entire timeseries. The `after` method is not necessary for this scheme and is left as the inherited empty method by default.

## Incremental Reservoirs
The `Incremental` abstract reservoir class processes data using multiple circuits over a moving substring of the timeseries. Circuits of length at most the specified `memory` parameter are created for step-by-step processing of the timeseries according to the circuit construction methods. Each circuit with a fixed-length memory is sent to the Qiskit backend to be run with the remaining parameters to the processing methods. Using this scheme, measurements taken in the `after` method can be post-processed to provide the user with the desired reservoir output.

An example of an `Incremental` quantum reservoir is the following.

```python
class QuantumReservoir(Incremental):
    def before(self, circuit):
        circuit.h(circuit.qubits)

    def during(self, circuit, timestep):
        circuit.initialize(encoder[timestep], [0, 1])
        circuit.append(operator, circuit.qubits)

    def after(self, circuit):
        circuit.measure_all()

```

In the example, the `during` method is still used to encode the current timestep in the timeseries, but measurement is instead taken in the `after` method. Since a circuit is created for each timestep, the combined measurements produce the desired sequential data.

## Custom Reservoirs

Custom reservoirs provide full flexibility over processing data in a quantum reservoir. A custom reservoir can be created by implementing the `QReservoir` abstract class. Unlike the prepackaged reservoirs, a custom reservoir must implement the `run` and `predict` methods, in addition to the circuit construction methods.

## Processing

When a quantum reservoir is instantiated, it requires a Qiskit backend. If no backend is specified, then `AerSimulator` is used by default. The backend must support the circuit operations specified in the circuit construction methods. When creating a custom reservoir, the `run` and `predict` methods should run the quantum reservoir on `self.backend` attribute.

As an example, instantiation of a quantum reservoir using the simulator backend `FakeTorontoV2` is as follows.
```python
from qiskit.providers.fake_provider import FakeTorontoV2

backend = FakeTorontoV2()
reservoir = QuantumReservoir(n_qubits=4, backend=backend)
```


Once a quantum reservoir has been instantiated, training data for the trainable model is produced by using the `run` method on the timeseries. The trained model can then be passed to the `predict` method to make predictions:

```python
output = reservoir.run(timeseries=timeseries, shots=10000)
# ...(model training)...
predictions = reservoir.predict(num_pred=10, model=model, from_series=timeseries, shots=10000)
```
Model training of a scikit-learn estimator between the `run` and `predict` methods is not shown. The `shots` parameter is directly passed to the Qiskit backend when using a prepackaged reservoir.

## Example
An example for how a quantum reservoir can be used for predictions is given in Figure~\ref{fig:simple-overview}.
A \texttt{Static} quantum reservoir with four qubits is used to predict a discrete sequence of zeros and ones.
The backend is an ideal `AerSimulator`.
One can see that the reservoir readout can be differentiated with linear regression, which leads to very good predictions.

# Conclusion and Further Development
We have introduced a new software package to encapsulate the possibilities of quantum reservoir computing within a succinct and easy-to-use framework. The structure has been designed to provide the full suite of the Qiskit circuit library when choosing a reservoir layout involving timeseries encoding, reservoir dynamics, and qubit measurement. Post-processing on decoded measurements and prediction generation are provided as built-in and customizable features for the implementation of a quantum reservoir. Prepackaged quantum reservoirs with common processing schemes are included.

The authors continue to support and maintain the project. Users may report package issues and desired features by opening an issue on the public GitHub repository or contacting the authors by email. Additional opportunities for further development on `QuantumReservoirPy` include supplementary built-in processing schemes, expanded features for data visualization, and reservoir evaluation methods.
# Acknowledgements
Work in this project was supported by the NTNU and SINTEF Digital through the International Work-Integrated-Learning in Artificial Intelligence (IWIL AI) program, in partnership with SFI NorwAI and the Waterloo Artificial Intelligence Institute (Waterloo.AI). IWIL AI is funded by the Norwegian Directorate for Higher Education and Skills (HK-dir).



# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }


# References
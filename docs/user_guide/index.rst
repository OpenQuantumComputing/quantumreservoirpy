.. _user_guide:

==========
User Guide
==========

Installation
============

QuantumReservoirPy is available as a pip-installable package on |ref_pypi|.

.. TODO: The link below should be updated to the actual package page on PyPI once it is published.

.. |ref_pypi| raw:: html

   <a href="https://pypi.org/" target="_blank">PyPI</a>


.. code-block:: console

   pip install qreservoirpy


This will install QuantumReservoirPy along with its dependent packages.

Alternative Options
-------------------

Alternatively, you may clone the repository from GitHub

.. code-block:: console

   git clone https://github.com/OpenQuantumComputing/quantumreservoirpy.git


navigate to the ``qreservoirpy`` directory

.. code-block:: console

   cd quantumreservoirpy/qreservoirpy


and install it from here.

.. code-block:: console

   pip install .


This is functionally equivalent to installing from PyPI.

Getting Started
===============

The interface of this package is somewhat inspired by |ref_reservoirpy|. Consider checking out |ref_tutorials| to better understand this package.

.. |ref_reservoirpy| raw:: html

   <a href="https://reservoirpy.readthedocs.io/" target="_blank">ReservoirPy</a>


.. |ref_tutorials| raw:: html

   <a href="https://reservoirpy.readthedocs.io/en/latest/user_guide/index.html" target="_blank">their tutorials</a>


Interface
---------

A *reservoir* is in this package defined as a class implementing the abstract class ``QReservoir``. To create a completely custom reservoir, you need to implement 5 functions as shown.

.. code-block:: python

   class CustomRes(QReservoir):
      def before(self, circuit):
         pass
      def during(self, circuit, timestep):
         pass
      def after(self, circuit):
         pass

      def run(self, timeseries, **kwargs):
         pass
      def predict(self, num_pred, model, from_series, **kwargs):
         pass

QuantumReservoirPy has some partially implemented reservoirs already, which have easier interfaces.

Static and Incremental
----------------------

The ``Static`` and ``Incremental`` reservoirs have implemented the ``run`` and ``predict`` methods, so you only need to implement ``before``, ``during``, and ``after``.

All the reservoirs created with ``Static`` and ``Ìncremental`` have the same three layered circuit structure; they begin with an initialization, which is defined by ``before``. Then, a small circuit is created for every timestep in the timeseries, which is defined by ``during``. The third and last layer is defined by ``after``.

.. code-block:: python

   from qreservoirpy.reservoirs import Static
   
   class CustomRes(Static):
      def before(self, circuit):
         circuit.h(circuit.qubits)
         circuit.barrier()

      def during(self, circuit, timestep):
         circuit.initialize(str(timestep), [0])
         circuit.h(0)
         circuit.cx(0, 1)

      def after(self, circuit):
         circuit.barrier()
         circuit.measure_all()

   res = CustomRes(n_qubits=2)
   res.circuit([0, 1]).draw('mpl')


.. TODO: Add the same image in README.md

The three methods ``before``, ``during``, and ``after`` do the same thing for both ``Static`` and ``Ìncremental`` reservoirs. The difference between them is what happens when the reservoirs are run.

Running a Reservoir
-------------------

Having created a reservoir, you can simply call ``reservoir.run``.

.. code-block:: python

   states = res.run(timeseries)


This will return a ``np.ndarray`` of the same length as the timeseries, corresponding to the reservoir state at each timestep.

``Static`` reservoirs run once and all measurements are reshaped to a ``(len(timeseries), -1)`` shape.

``Incremental`` reservoirs run incrementally. For every state, only the last ``M`` steps of the timeseries is built at a time (``M`` being a parameter of ``Incremental.__init__``).

Examples
========

There are several examples available in the GitHub repository.

* |ref_qubit|
* |ref_qubit_longer|
* |ref_clifford|
* |ref_random_unitary|
* |ref_twinkle|

.. |ref_qubit| raw:: html

   <a href="https://github.com/OpenQuantumComputing/quantumreservoirpy/blob/main/examples/static/1Qbit.ipynb" target="_blank">1 Qubit (Static)</a>


.. |ref_qubit_longer| raw:: html

   <a href="https://github.com/OpenQuantumComputing/quantumreservoirpy/blob/main/examples/static/1Qbit_longer_sequence.ipynb" target="_blank">1 Qubit, Longer Sequence (Static)</a>


.. |ref_clifford| raw:: html

   <a href="https://github.com/OpenQuantumComputing/quantumreservoirpy/blob/main/examples/static/clifford.ipynb" target="_blank">Clifford (Static)</a>


.. |ref_random_unitary| raw:: html

   <a href="https://github.com/OpenQuantumComputing/quantumreservoirpy/blob/main/examples/incremental/randomunitary.ipynb" target="_blank">Random Unitary (Incremental)</a>


.. |ref_twinkle| raw:: html

   <a href="https://github.com/OpenQuantumComputing/quantumreservoirpy/blob/main/examples/music/twinkle.ipynb" target="_blank">Twinkle (Incremental)</a>


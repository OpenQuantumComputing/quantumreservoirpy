.. _api_reference:

=============
API Reference
=============

Reservoirs
==========

qreservoirpy.reservoirs.QReservoir
----------------------------------

*Abstract Class*

.. code-block:: python

    qreservoirpy.reservoirs.QReservoir(n_qubits,  memory=np.inf, backend=None)
    

Reservoir that allows for full customization of all methods.

   =======================================================  ==============================================================
   Implementable Methods
   =======================================================  ==============================================================
   before(self, circuit)                                    Specifies how to initialize the circuit.
   during(self, circuit, timestep)                          Specifies each timestep in the circuit.
   after(self, circuit)                                     Specifies the final layer after every timestep in the circuit.
   run(self, timeseries, \*\*kwargs)                        Runs the timeseries on the reservoir using the backend.
   predict(self, num_pred, model, from_series, \*\*kwargs)  Makes predictions using the reservoir and given model.
   =======================================================  ==============================================================


qreservoirpy.reservoirs.Static
------------------------------

*Class*

.. code-block:: python

   qreservoirpy.reservoir.Static(n_qubits,  memory=np.inf, backend=None)


Reservoir which runs once on the timeseries. All measurements are reshaped to ``(len(timeseries), -1)``.

   =======================================================  ==============================================================
   Implementable Methods
   =======================================================  ==============================================================
   before(self, circuit)                                    See :ref:`qreservoirpy.reservoirs.QReservoir`.
   during(self, circuit, timestep)                          See :ref:`qreservoirpy.reservoirs.QReservoir`.
   after(self, circuit)                                     See :ref:`qreservoirpy.reservoirs.QReservoir`.
   =======================================================  ==============================================================


qreservoirpy.reservoirs.Incremental
-----------------------------------

*Class*

.. code-block:: python

   qreservoirpy.reservoir.Incremental(n_qubits, memory=np.inf, backend=None, num_features=-1)


Reservoir which runs incrementally on the timeseries, where only the last ``M`` steps of the timeseries is built for every state.

   =======================================================  ==============================================================
   Implementable Methods
   =======================================================  ==============================================================
   before(self, circuit)                                    See :ref:`qreservoirpy.reservoirs.QReservoir`.
   during(self, circuit, timestep)                          See :ref:`qreservoirpy.reservoirs.QReservoir`.
   after(self, circuit)                                     See :ref:`qreservoirpy.reservoirs.QReservoir`.
   =======================================================  ==============================================================


Plotting
========

qreservoirpy.plot.state_plotter
-------------------------------

*Function*

.. code-block:: python

   qreservoirpy.plot.state_plotter(x, target)


Music
=====

qreservoirpy.music.gen_audio
----------------------------

*Function*

.. code-block:: python

   qreservoirpy.music.gen_audio(noter, filename="output.wav", BPM=144)


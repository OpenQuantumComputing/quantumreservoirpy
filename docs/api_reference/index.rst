.. _api_reference:

=============
API Reference
=============

Reservoirs
==========

quantumreservoirpy.reservoirs.QReservoir
----------------------------------------

*Abstract Class*

.. code-block:: python

    quantumreservoirpy.reservoirs.QReservoir(n_qubits,  memory=np.inf, backend=None)
    

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


quantumreservoirpy.reservoirs.Static
------------------------------------

*Class*

.. code-block:: python

   quantumreservoirpy.reservoir.Static(n_qubits,  memory=np.inf, backend=None)


Reservoir which runs once on the timeseries. All measurements are reshaped to ``(len(timeseries), -1)``.

   =======================================================  ==============================================================
   Implementable Methods
   =======================================================  ==============================================================
   before(self, circuit)                                    See :ref:`quantumreservoirpy.reservoirs.QReservoir`.
   during(self, circuit, timestep)                          See :ref:`quantumreservoirpy.reservoirs.QReservoir`.
   after(self, circuit)                                     See :ref:`quantumreservoirpy.reservoirs.QReservoir`.
   =======================================================  ==============================================================


quantumreservoirpy.reservoirs.Incremental
-----------------------------------------

*Class*

.. code-block:: python

   quantumreservoirpy.reservoir.Incremental(n_qubits, memory=np.inf, backend=None, num_features=-1)


Reservoir which runs incrementally on the timeseries, where only the last ``M`` steps of the timeseries is built for every state.

   =======================================================  ==============================================================
   Implementable Methods
   =======================================================  ==============================================================
   before(self, circuit)                                    See :ref:`quantumreservoirpy.reservoirs.QReservoir`.
   during(self, circuit, timestep)                          See :ref:`quantumreservoirpy.reservoirs.QReservoir`.
   after(self, circuit)                                     See :ref:`quantumreservoirpy.reservoirs.QReservoir`.
   =======================================================  ==============================================================


Plotting
========

quantumreservoirpy.plot.feature_plotter
---------------------------------------

*Function*

.. code-block:: python

   quantumreservoirpy.plot.feature_plotter(x, target)


Plots two reservoir features in a single scatterplot. Defaults to the first two features of ``x``.


quantumreservoirpy.plot.state_plotter
-------------------------------------

*Function*

.. code-block:: python

   quantumreservoirpy.plot.state_plotter(x, target)


Plots each reservoir feature in individual side-by-side scatterplots.


Music
=====

quantumreservoirpy.music.gen_audio
----------------------------------

*Function*

.. code-block:: python

   quantumreservoirpy.music.gen_audio(noter, filename="output.wav", BPM=144)


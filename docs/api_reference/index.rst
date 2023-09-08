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
    

qreservoirpy.reservoirs.Static
------------------------------

*Class*

.. code-block:: python

   qreservoirpy.reservoir.Static(n_qubits,  memory=np.inf, backend=None)


qreservoirpy.reservoirs.Incremental
-----------------------------------

*Class*

.. code-block:: python

   qreservoirpy.reservoir.Incremental(n_qubits, memory=np.inf, backend=None, num_features=-1)


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


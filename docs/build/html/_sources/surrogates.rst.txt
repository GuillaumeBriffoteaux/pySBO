Surrogates
==========

.. automodule:: Surrogates.__init__

Classes summary
---------------
.. autosummary::

   Surrogates.Surrogate.Surrogate
   Surrogates.BNN_MCD.BNN_MCD
   Surrogates.BLR_ANN.BLR_ANN
   Surrogates.iKRG.iKRG
   Surrogates.rKRG.rKRG
   Surrogates.GP.GP
   Surrogates.GP_MO.GP_MO
   
Surrogate (abstract)
--------------------
.. autoclass:: Surrogates.Surrogate.Surrogate


Approximated Bayesian Neural Network
------------------------------------

Monte Carlo Dropout
^^^^^^^^^^^^^^^^^^^
.. autoclass:: Surrogates.BNN_MCD.BNN_MCD

	       
Bayesian Linear Regressor
-------------------------
	       
Artificial Neural Network basis functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Surrogates.BLR_ANN.BLR_ANN


Gaussian Processes
------------------

Interpolation Kriging
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Surrogates.iKRG.iKRG

Regression Kriging
^^^^^^^^^^^^^^^^^^
.. autoclass:: Surrogates.rKRG.rKRG
	       
GP with different kernels
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Surrogates.GP.GP

GP for multiple ojbectives
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Surrogates.GP_MO.GP_MO

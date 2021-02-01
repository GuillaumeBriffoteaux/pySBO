Surrogates
==========

.. automodule:: Surrogates.__init__

Classes summary
---------------
.. autosummary::

   Surrogates.Surrogate.Surrogate
   Surrogates.BNN_MCD.BNN_MCD
   Surrogates.BNN_BLR.BNN_BLR
   Surrogates.KRG.KRG
   Surrogates.GP_Matern.GP_Matern
   Surrogates.GP_SMK.GP_SMK
   Surrogates.RF.RF
   
Surrogate (abstract)
--------------------
.. autoclass:: Surrogates.Surrogate.Surrogate


Approximated Bayesian Neural Network
------------------------------------

Monte Carlo Dropout
^^^^^^^^^^^^^^^^^^^
.. autoclass:: Surrogates.BNN_MCD.BNN_MCD

Bayesian Linear Regressor
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Surrogates.BNN_BLR.BNN_BLR


Gaussian Processes
------------------

Kriging
^^^^^^^
.. autoclass:: Surrogates.KRG.KRG

Matern kernel
^^^^^^^^^^^^^
.. autoclass:: Surrogates.GP_Matern.GP_Matern

Spectral Mixture kernel
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Surrogates.GP_SMK.GP_SMK
	       
Others
------

Random Forest
^^^^^^^^^^^^^
.. autoclass:: Surrogates.RF.RF

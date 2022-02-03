Evolution Controls
==================

.. automodule:: Evolution_Controls.__init__

Classes summary
---------------
.. autosummary::

   Evolution_Controls.Evolution_Control.Evolution_Control
   Evolution_Controls.Random_EC.Random_EC
   Evolution_Controls.Informed_EC.Informed_EC
   Evolution_Controls.POV_EC.POV_EC
   Evolution_Controls.Distance_EC.Distance_EC
   Evolution_Controls.Pred_Stdev_EC.Pred_Stdev_EC
   Evolution_Controls.Expected_Improvement_EC.Expected_Improvement_EC
   Evolution_Controls.Probability_Improvement_EC.Probability_Improvement_EC
   Evolution_Controls.Lower_Confident_Bound_EC.Lower_Confident_Bound_EC
   Evolution_Controls.Adaptive_Wang2020_EC.Adaptive_Wang2020_EC
   Evolution_Controls.MO_POV_LCB_EC.MO_POV_LCB_EC
   Evolution_Controls.MO_POV_LCB_IC.MO_POV_LCB_IC
   Evolution_Controls.Ensemble_EC.Ensemble_EC
   Evolution_Controls.Pareto_EC.Pareto_EC
   Evolution_Controls.Pareto_Tian2018_EC.Pareto_Tian2018_EC
   Evolution_Controls.Dynamic_Exclusive_EC.Dynamic_Exclusive_EC
   Evolution_Controls.Dynamic_Inclusive_EC.Dynamic_Inclusive_EC
   Evolution_Controls.Adaptive_EC.Adaptive_EC
   Evolution_Controls.Committee_EC.Committee_EC

Naive
-----

Evolution Control (abstract)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Evolution_Controls.Evolution_Control.Evolution_Control

Random
^^^^^^
.. autoclass:: Evolution_Controls.Random_EC.Random_EC


Informed
--------
	       
Informed (abstract)
^^^^^^^^^^^^^^^^^^^
.. autoclass:: Evolution_Controls.Informed_EC.Informed_EC

Predicted Objective Value
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Evolution_Controls.POV_EC.POV_EC

Distance
^^^^^^^^
.. autoclass:: Evolution_Controls.Distance_EC.Distance_EC

Predictive Standard Deviation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Evolution_Controls.Pred_Stdev_EC.Pred_Stdev_EC

Expected Improvement
^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Evolution_Controls.Expected_Improvement_EC.Expected_Improvement_EC
	       
Probability Improvement
^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Evolution_Controls.Probability_Improvement_EC.Probability_Improvement_EC

Lower Confident Bound
^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Evolution_Controls.Lower_Confident_Bound_EC.Lower_Confident_Bound_EC

Adaptive (from Wang-2020)
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Evolution_Controls.Adaptive_Wang2020_EC.Adaptive_Wang2020_EC


Multi-objective POV and LCB (from Ruan-2020)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Evolution_Controls.MO_POV_LCB_EC.MO_POV_LCB_EC
.. autoclass:: Evolution_Controls.MO_POV_LCB_IC.MO_POV_LCB_IC


Ensemble
--------
	       
Ensemble (abstract)
^^^^^^^^^^^^^^^^^^^
.. autoclass:: Evolution_Controls.Ensemble_EC.Ensemble_EC

Pareto
^^^^^^
.. autoclass:: Evolution_Controls.Pareto_EC.Pareto_EC

Pareto-based (from Tian-2018)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Evolution_Controls.Pareto_Tian2018_EC.Pareto_Tian2018_EC
	       
Dynamic Exclusive
^^^^^^^^^^^^^^^^^
.. autoclass:: Evolution_Controls.Dynamic_Exclusive_EC.Dynamic_Exclusive_EC

Dynamic Inclusive
^^^^^^^^^^^^^^^^^
.. autoclass:: Evolution_Controls.Dynamic_Inclusive_EC.Dynamic_Inclusive_EC	       

Adaptive
^^^^^^^^
.. autoclass:: Evolution_Controls.Adaptive_EC.Adaptive_EC

Committee
^^^^^^^^^
.. autoclass:: Evolution_Controls.Committee_EC.Committee_EC

About pySBO
===========

The platform aims at facilitating the implementation of parallel surrogate-based optimization algorithms. pySBO provides re-usable algorithmic components (surrogate models, evolution controls, infill criteria, evolutionary operators) as well as the foundations to ensure the components inter-changeability. Actual implementations of sequential and parallel surrogate-based optimization algorithms are supplied as ready-to-use tools to handle expensive single- and multi-objective problems with continuous decision variables. The MPI implementation of parallel algorithms allows to execute on distributed machines. Box-constraints are explicitly integrated while more elaborated constraints must be handled by the user.


Installation
------------

Clone the Github repository from `<https://github.com/GuillaumeBriffoteaux/pySBO.git>`_

Check `python3.9` is installed on your system.

Install the required libraires: ``python3.9 -m pip install -r requirements.txt``

Requirement file :download:`available here<../../requirements.txt>`


Notes
-----

pySBO is organized following the one-class-per-file Java convention. Consequently, each module is nammed after the class it contains.


Support
-------

guillaume.briffoteaux@gmail.com


Background on Surrogate-Based Optimization
------------------------------------------

`G. Briffoteaux, R. Ragonnet, M. Mezmaz, N. Melab and D. Tuyttens. Evolution Control Ensemble Models for Surrogate-Assisted Evolutionary Algorithms. HPCS 2020 - International Conference on High Performance Computing and Simulation, 22-27 March 2021, Onlineconference. <https://hal.inria.fr/hal-03332521>`_

`G.Briffoteaux, M.Gobert, R.Ragonnet, J.Gmys, M.Mezmaz, N.Melab and D.Tuyttens. Parallel Surrogate-assisted Optimization: Batched Bayesian Neural Network-assisted GA versus q-EGO. Swarm and Evolutionary Computation , 57:100717, 2020. <https://www.sciencedirect.com/science/article/abs/pii/S2210650220303709?via%3Dihub>`_

`G.Briffoteaux, R.Ragonnet, M.Mezmaz, N.Melab and D.Tuyttens. Evolution Control for Parallel ANN-assisted Simulation-based Optimization, Application to Tuberculosis Transmission Control. Future Generation Computer System , 113:454-467, 2020. <https://www.sciencedirect.com/science/article/abs/pii/S0167739X19308635>`_


Author
------

`Guillaume Briffoteaux <https://www.linkedin.com/in/gbriffoteaux/>`_


License
-------

:download:`Available here<../../LICENSE.txt>`


Supporting institutions
-----------------------

Faculté Polytech Mons, Université de Mons, Belgique

.. image:: ../logos/logo_polytech.jpeg
	   :scale: 75%
	   :align: center

.. image:: ../logos/logo_umons.png
	   :scale: 75%
	   :align: center		   

Université de Lille, CNRS CRIStAL, Inria Lille, France

.. image:: ../logos/logo_lille.png
	   :scale: 50%
	   :align: center

.. image:: ../logos/logo_cnrs.png
	   :scale: 30%
	   :align: center

.. image:: ../logos/logo_cristal.png
	   :scale: 40%
	   :align: center

.. image:: ../logos/logo_inria.png
	   :scale: 10%
	   :align: center


Collaborators
-------------
		   
School of Public Health and Preventive Medicine, Monash University, Australia

.. image:: ../logos/logo_monash.png
	   :scale: 30%
	   :align: center

About pySBO
===========

The platform aims at facilitating the implementation of parallel surrogate-based optimization algorithms. pySBO provides re-usable algorithmic components (surrogate models, evolution controls, infill criteria, evolutionary operators) as well as the foundations to ensure the components inter-changeability. Actual implementations of sequential and parallel, surrogate-based and surrogate-free optimization algorithms are supplied as ready-to-use tools to handle very and moderately expensive single- and multi-objective problems with continuous decision variables. The MPI implementation allows to execute on distributed machines. Box-constraints are explicitly integrated while more elaborated constraints must be handled by the user.

At a glance:

* Surrogate-Assisted Evolutionary Algorithms for moderately expensive problems
* Surrogate-Driven Algorithms for very expensive problems
* Surrogate-Free Algorithms for unexpensive problems
* Single- and multi-objective
* Continuous decision variables
* Parallel evaluations of the objective function
* Parallel Acquisition Processes
* Centered on Evolutionary Algorithms
* Box-constrained problems


Notes
-----

pySBO is organized following the one-class-per-file Java convention. Consequently, each module is nammed after the class it contains.


Support
-------

guillaume.briffoteaux@gmail.com


Background on Surrogate-Based Optimization
------------------------------------------

`G. Briffoteaux. Parallel surrogate-based algorithms for solving expensive optimization problems. Thesis. University of Mons (Belgium) and University of Lille (France). 2022. <https://hal.science/tel-03853862>`_

`G. Briffoteaux, N. Melab, M. Mezmaz et D. Tuyttens. Hybrid Acquisition Processes in Surrogate-based Optimization. Application to Covid-19 Contact Reduction. International Conference on Bioinspired Optimisation Methods and Their Applications, BIOMA, 2022, Maribor, Slovenia, Lecture Notes in Computer Science, vol 13627. Springer, pages 127-141 <https://doi.org/10.1007/978-3-031-21094-5_10>`_

`G. Briffoteaux, R. Ragonnet, P. Tomenko, M. Mezmaz, N. Melab et D. Tuyttens. Comparing Parallel Surrogate-based and Surrogate-free Multi-Objective Optimization of COVID-19 vaccines allocation. International Conference on Optimization and Learning, OLA, 2022, Syracuse, Italy, Communications in Computer and Information Science, vol 1684. Springer, pages 201-212, <https://doi.org/10.1007/978-3-031-22039-5_16>`_

`G. Briffoteaux, R. Ragonnet, M. Mezmaz, N. Melab and D. Tuyttens. Evolution Control Ensemble Models for Surrogate-Assisted Evolutionary Algorithms. HPCS 2020 - International Conference on High Performance Computing and Simulation, 22-27 March 2021, Onlineconference. <https://hal.inria.fr/hal-03332521>`_

`G.Briffoteaux, M.Gobert, R.Ragonnet, J.Gmys, M.Mezmaz, N.Melab and D.Tuyttens. Parallel Surrogate-assisted Optimization: Batched Bayesian Neural Network-assisted GA versus q-EGO. Swarm and Evolutionary Computation, 57:100717, 2020. <https://www.sciencedirect.com/science/article/abs/pii/S2210650220303709?via%3Dihub>`_

`G.Briffoteaux, R.Ragonnet, M.Mezmaz, N.Melab and D.Tuyttens. Evolution Control for Parallel ANN-assisted Simulation-based Optimization, Application to Tuberculosis Transmission Control. Future Generation Computer System, 113:454-467, 2020. <https://www.sciencedirect.com/science/article/abs/pii/S0167739X19308635>`_

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

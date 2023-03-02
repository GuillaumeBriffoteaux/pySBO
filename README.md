The platform aims at facilitating the implementation of parallel surrogate-based optimization algorithms. pySBO provides re-usable algorithmic components (surrogate models, evolution controls, infill criteria, evolutionary operators) as well as the foundations to ensure the components inter-changeability. Actual implementations of sequential and parallel, surrogate-based and surrogate-free optimization algorithms are supplied as ready-to-use tools to handle very and moderately expensive single- and multi-objective problems with continuous decision variables. The MPI implementation allows to execute on distributed machines. Box-constraints are explicitly integrated while more elaborated constraints must be handled by the user.

pySBO is organized following the one-class-per-file Java convention. Consequently, each module is nammed after the class it contains.

At a glance:
   - Surrogate-Assisted Evolutionary Algorithms for moderately expensive problems
   - Surrogate-Driven Algorithms for very expensive problems
   - Surrogate-Free Algorithms for unexpensive problems
   - Single- and multi-objective
   - Continuous decision variables
   - Parallel evaluations of the objective function
   - Parallel Acquisition Processes
   - Centered on Evolutionary Algorithms
   - Box-constrained problems

See the documentation at https://pysbo.readthedocs.io
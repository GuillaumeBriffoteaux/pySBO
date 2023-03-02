Installation
============
  

Linux
-----

This procedure has been tested on Ubuntu20.04 and on Debian11.

1.  Open a terminal.

2.  Download Miniconda by running::
      
      $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

3.  Install Miniconda by running::
      
      $ bash Miniconda3-latest-Linux-x86_64.sh
      
    * Accept the license term.
    * Accept the installation folder.
    * Accept to initialize Miniconda3.
      
    Close the terminal and open a new one. `(base)` should appear at the beginning of the terminal line.

    To prevent conda's base environment to be activated when a new terminal is open::
      
      $ conda config --set auto_activate_base false

4.  Create a new Python3.9 environment thanks to Miniconda3::

      $ conda create --name pySBOenv python=3.9

5.  Activate the new environment::
      
      $ conda activate pySBOenv
      
    When `pySBOenv` is activated, `(pySBOenv)` appears at the beginning of a terminal line.

6.  Install Pygmo(>=2.19.0)::
      
      (pySBOenv) $ conda config --add channels conda-forge
      (pySBOenv) $ conda config --set channel_priority strict
      (pySBOenv) $ conda install pygmo

7.  Install the other dependencies, either one-by-one::
      
      (pySBOenv) $ pip install numpy>=1.24.2
      (pySBOenv) $ pip install mpi4py>=3.1.4
      (pySBOenv) $ pip install matplotlib>=3.7.0
      (pySBOenv) $ pip install pybnn>=0.0.5
      (pySBOenv) $ pip install gpytorch>=1.9.1
      (pySBOenv) $ pip install tensorflow-cpu>=2.11.0
      (pySBOenv) $ pip install scipy>=1.10.1
      (pySBOenv) $ pip install pyDOE>=0.3.8
      (pySBOenv) $ pip install pyKriging>=0.2.0
      (pySBOenv) $ pip install scikit_learn>=1.2.1
      (pySBOenv) $ pip install pyro-ppl>=1.8.4

    or by retrieving the requirement file :download:`available here<../../requirements.txt>` and running::

      (pySBOenv) $ pip install -r requirements.txt


8.  Check the dependencies have been properly installed by importing them in Python::
      
      (pySBOenv) $ python
      >>> import pygmo, numpy, mpi4py, matplotlib, pybnn, gpytorch
      >>> import tensorflow, scipy, pyDOE, pyKriging, sklearn, pyro
      
    If some error related to `GLIBCXX` version shows up, update `libstdc++.so.6` by running::
      
      (pySBOenv) $ sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
      (pySBOenv) $ sudo apt update
      (pySBOenv) $ sudo apt upgrade libstdc++6

9.  Download `pySBO` by running::

      (pySBOenv) $ wget https://github.com/GuillaumeBriffoteaux/pySBO/archive/refs/heads/main.zip

10. Extract from the archive::

      (pySBOenv) $ unzip main.zip

11. To test the installation, go to the `pySBO-main/examples` directory and run the parallel Surrogate-Assisted Evolutionary Algorithm::
      
      (pySBOenv) $ mpiexec -n 2 python SAEA.py
      
    Check the `outputs` directory.


Windows
-------

This procedures has been tested on Windows10.

1.  Install Miniconda3 by following the instructions given at:

    `<https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_
    
    Once Miniconda3 is installed, open `Anaconda Prompt (miniconda3)`.

2.  Deactivate the `base` environment::
      
      (base) > conda deactivate

3.  Create a new Python3.9 environment::
      
      > conda create --name pySBOenv python=3.9

4.  Activate the new environment::

      > conda activate pySBOenv
      
    When `pySBOenv` is activated, `(pySBOenv)` appears at the beginning of the prompt line.

5.  Install Pygmo(>=2.19.0)::
      
      (pySBOenv) > conda config --add channels conda-forge
      (pySBOenv) > conda config --set channel_priority strict
      (pySBOenv) > conda install pygmo

6.  Install the other dependencies::
      
      (pySBOenv) > pip install numpy>=1.24.2
      (pySBOenv) > pip install mpi4py>=3.1.4
      (pySBOenv) > pip install matplotlib>=3.7.0
      (pySBOenv) > pip install pybnn>=0.0.5
      (pySBOenv) > pip install gpytorch>=1.9.1
      (pySBOenv) > pip install tensorflow-cpu>=2.11.0
      (pySBOenv) > pip install scipy>=1.10.1
      (pySBOenv) > pip install pyDOE>=0.3.8
      (pySBOenv) > pip install pyKriging>=0.2.0
      (pySBOenv) > pip install scikit_learn>=1.2.1
      (pySBOenv) $ pip install pyro-ppl>=1.8.4

7.  Check the dependencies have been properly installed by importing them in Python::

      (pySBOenv) > python
      >>> import pygmo, numpy, mpi4py, matplotlib, pybnn, gpytorch, tensorflow, scipy
      >>> import pyDOE, pyKriging, sklearn, pyro

8.  Download `MS MPI` from:

    `<https://www.microsoft.com/en-us/download/details.aspx?id=57467>`_
    
    and install it (by running `msmpisetup.exe`).

9.  Add the `MS MPI` bin folder (by default `C:\\Program Files\\Microsoft MPI\\Bin\\`) to the `%PATH%` environment variable. Follow the following tutorial to edit the `%PATH%` environment variable in Windows.

    `<https://www.computerhope.com/issues/ch000549.htm>`_

10. Download `pySBO` from

    `<https://github.com/GuillaumeBriffoteaux/pySBO/archive/refs/heads/main.zip>`_
    
    and extract from the archive.

11. To test the installation, go to the `pySBO-main/examples` directory from the `Anaconda Prompt (miniconda3)`. Then run::
      
      (pySBOenv) > mpiexec /np 2 python SAEA.py

    Check the `outputs` directory.

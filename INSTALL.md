Installation
===

The code is quite easy to use it only requires a proper Python3 environment. Therefore, 
if not already installed, make sure you have python3 installed

On linux this can simply be done by using the apt-get command. For example for Python3.8:

```console
  sudo apt-get install python3.8
```
Make sure you have sudo permissions otherwise the installation will not work.

Pip
------------------------------------------

Pip is a python package which helps you to install all additional packages needed to run the code.
It is probably the most easiest way to install all necessary dependencies. Pip is included by default
in Python3.4 and later versions.

For the latest scikit-learn version simply use:

```console
  pip install -U scikit-learn
```
All other packages can be installed in the same way:
```console
  pip install -U scikit-optimize
```
```console
  pip install -U scikit-image
```
```console
  pip install -U numpy
```
```console
  pip install -U scipy
```
```console
  pip install -U matplotlib
```
For a more detailed information on how to install scikit-learn
please visit https://scikit-learn.org/stable/install.html

Installation with Package manager
------------------------------------------
On Debian/Ubuntu scikit-learn is split in three different packages

```console
  sudo apt-get install python3-sklearn python3-sklearn-lib python3-sklearn-doc
```
By running this command necessary dependencies (numpy/scipy) will be automatically installed.

If necessary, with 
```console
  sudo apt-get install python3-matplotlib
```
you can install also matplotlib.

You can now run BYND as described in the example section.




# Biological Network BackProp

# Components & Requirements
## Neural Network simulation & training
Simulates and trains a network of neurons to learn the MNIST dataset. Also performs preprocessing of MNIST dataset.  
Requirements:
 - the `make` build tool
 - gcc or any other C++ compiler
 - openMP (included with gcc)
 - Eigen (included in repo) ## TODO
 - Boost (not included in repo)

## Parameter sweeping
Searches the hyperparameter space
Requirements:
 - python 3.7 or greater
 - numpy
 - nevergrad
 - python-fire


## Plotting scripts
matlab and python scripts for generating figures
(TODO: convert matlab scripts to python and organize it all)
Requirements:
 - python 3.7 or greater
 - numpy
 - pandas
 - matplotlib
 - python-fire
 
# Timing

## Runtime Estimates

Plotting the figures should only take a matter of seconds. To run a full training session on the MNIST dataset can take a while, up to an hour, but running on a subset can take about 10 minutes. 

## Compile Time

Compiling usually only takes a matter of seconds or a few minutes. 




# using [python-fire](https://google.github.io/python-fire/)
- to see the list of functions availible in a module, do 
  ```python file_name.py --help```
- to get help about a specific function/command within a module, do
  ```python file_name.py COMMAND -- --help```
  (the standalone `--` acts as a separator)

# generating documentation
## python
uses pdoc3, install using `pip install pdoc3`  
to build documentation, run `pdoc --html --force . -o docs`



# NEED:
- [ ] small demo
- [ ] requirements
  - [ ] all dependecies, including version numbers
  - [ ] versions the software has been tested on
- [ ] installation guide
  - [ ] makefile
  - [ ] instructions
  - [ ] typicall install time
- [ ] demo
  - [ ] instructions to run
  - [ ] expected output
  - [ ] expected runtime
- [ ] general instructions for use

# 2021-03-18 01:26 todo:
- requirements.txt for pip and python things
- list of where Eigen and whatever other C++ packages we use
- runtime estimates (done?)
- explain structure of the output directories (!)
- plotting scripts are already documented, refer to them in the readme
- instructions for the launching scripts
- liscence
- link to repo







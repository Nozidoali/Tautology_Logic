# Tautology Logic with Reinforcement Learning Agent

## Install
First we install all the source code and the dependencies.
```
sudo apt-get install yosys
git clone --recursive https://github.com/Nozidoali/Tautology_Logic.git
```

Next, build the virtual environment for our program and install the dependencies:
```sh
cd Tautology_Logic
pip install virtualenv
virtualenv .venv --python=python3.6
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```
usage: main.py [-h] [--size] [--output OUTPUT] [--train TRAIN]

Tautology Generator

optional arguments:
  -h, --help       show this help message and exit
  --size           input number of the circuit
  --output OUTPUT  blif file directory
  --train TRAIN    blif file RL agent learn to optimize
```
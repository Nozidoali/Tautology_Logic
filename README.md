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
First, we use the output function to generate a tautology with 8 inputs:
```
python main.py --output output.blif --size 8
```

Then we train it using Reinforcement Learning
```
python main.py --train output.blif
```

Next, we retrieve the result using:
```
python main.py --print 50
```

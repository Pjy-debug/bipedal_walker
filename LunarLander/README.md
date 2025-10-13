# Intelligent testing environment for Lunar Lander

## Installation

First, install Python 3.8 and all the dependencies. Run:
```
pip install -r requirements.txt
```
Then, install the Lunar Lander environment. Run:
```
cd gym_testenvs
pip install -e .
```

## Process

First, following the three stages in https://arxiv.org/pdf/2403.13869 to construct criticality model (see folder ```Env_agent```).

Then, train the d2rl model (see folder ```d2rl_training```).

Finally, testing the Lunar Lander by ```nde_test.py```, ```nade_test.py```, and ```d2rl_test.py```, and comparing the results to calculate the acceleration ratio of testing.
# EMG experiments

This is the code for the paper *Signals to Spikes for Neuromorphic Regulated Reservoir Computing and EMG Hand Gesture Recognition*, available [here](https://arxiv.org/pdf/2106.11169.pdf).


# Getting started

## Dependencies
- numpy
- pandas
- matplotlib
- scipy
- scikit_learn
- scikit_plot
- nni
- seaborn
- Brian2

## Installation
Requirements can be installed with : 
```
pip3 install -r requirements.txt
```
  
Please note that the requirement file might use outdated dependencies. To get latest versions : 
```
pip3 install numpy pandas matplotlib scipy scikit_learn scikit_plot nni seaborn Brian2
```

Finally, make sure the source of the project is in the Python path.  
- On Linux : `export PYTHONPATH="$PWD/src"`
- On Windows : `set PYTHONPATH="$PWD/src"`

You can also modify it permanently : see [here](https://bic-berkeley.github.io/psych-214-fall-2016/using_pythonpath.html).


# Usage

## Datasets
Different datasets for Electromyography (EMG) are provided in the folder `datasets/` :
- Roshambo *(classes: rock, paper, scissor)*
- 5_class *(classes: pinky, elle, yo, index, thumb)*
- Pinch *(classes: Pinch1, Pinch2, Pinch3, Pinch4)*

## Pipelines
### Spike encoder & Evaluation baseline 
This is the first pipeline described in the paper. To execute it : 
```
python3 src/evaluate_encoder.py [options]
```
### Reservoir
This is the second pipeline described in the paper. To execute it :
```
python3 src/evaluate_reservoir.py [options] 
```

## Parameters
All the parameters are listed and detailed in the file `src/args_emg.py`. They can be specified as follow : 
```
python3 <<script>> --dataset="5_class" --learning_algorithm="critical" --cbf=1
```

## Scripting
Pipelines can also be used in Python scripts :
```python
import random
import numpy as np
from evaluate_encoder import *
from evaluate_reservoir import *
from args_emg import args as my_args

# Get default arguments 
# Can be modified (eg. args.adaptiveProb = 1)
args = my_args()

# Fix the seed of all random number generator
seed = int(args.seed)
random.seed(seed)
np.random.seed(seed)

# Spike encoding & Evaluation baseline
svm_score_enc,firing_rate_enc,svm_score_baseline_enc = evaluate_encoder(args)

# Reservoir 
lda_score,lda_score_input,svm_linear_score,svm_linear_score_input,svm_score,svm_score_input,firing_rate,nbsynapses,nbneurons = evaluate_reservoir(args)
```


# Reproduce the results

Experiments presented in the paper can be reproduced by executing the files in the folder `experiments/` : 
```
python3 -m experiments.experiment_exploration_v10
```

*Note that Spike encoder & Evaluation baseline is only used in the file `experiments/baseline_exploration.py`.*

## Best results 
### Spike encoder & Evaluation baseline
#### Roshambo dataset
TODO
#### 5_class
TODO
### Reservoir
#### Roshambo dataset
Default parameters reproduce the best results for Roshambo dataset : 
```
python3 -m src.evaluate_reservoir
```
#### 5_class
TODO


# Acknowledgements 
TODO.  
University of Sherbrooke. [NEuro COmputational & Intelligent Signal Processing Research Group (NECOTIS)](http://www.gel.usherbrooke.ca/necotis/)

<img src="data/necotis.png" width="250" /> <img src="data/UdeS.jpg" width="250" />

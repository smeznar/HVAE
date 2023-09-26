# Efficient Generator of Mathematical Expressions for Symbolic Regression

This repository contains code that is used and presented as part of the paper **_Efficient Generator of Mathematical Expressions for Symbolic Regression_**, that can be found [here](https://link.springer.com/article/10.1007/s10994-023-06400-2):

```
﻿@article{Mežnar2023HVAE,
  author={Me{\v{z}}nar, Sebastian and D{\v{z}}eroski, Sa{\v{s}}o and Todorovski, Ljup{\v{c}}o},
  title={Efficient generator of mathematical expressions for symbolic regression},
  journal={Machine Learning},
  year={2023},
  month={Sep},
  day={06},
  issn={1573-0565},
  doi={10.1007/s10994-023-06400-2},
  url={https://doi.org/10.1007/s10994-023-06400-2}
}
```

EDHiE, a mad scientist frantically searching for the right mathematical expression:

![edhie logo](https://github.com/smeznar/HVAE/blob/master/images/edhie1.jpeg)

**We are currently refactoring and improving some code and its performance, so some parts of the code might be broken.**

An overview of the approach (shown on the symbolic regression example) can be seen below.
![algorithm overview](https://github.com/smeznar/HVAE/blob/master/images/overview.png)

## Installing HVAE
To install and test HVAE, do the following:
  1. Install rust (instructions at [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install))
  2. Create a new (conda) environment
  3. Install dependencies with the command: `pip install -r requirements.txt`

## Using HVAE
TBA

## Evaluation scenarios
Our motivation for this approach is symbolic regression (equation discovery), a machine learning task where you try to find a closed-form solution that fits the given data.
In case of symbolic regression, HVAE is used to generate expression. To explore the latent space produced by HVAE efficiently, 
our variational autoencoder needs to possess the following characteristics:
- Produce syntactically valid expressions; HVAE produces only syntactically valid expressions by design,
- Reconstruct (unseen) expressions well; otherwise we cannot expect that the latent space will have structure and the expressions
produced by the generator are always random (we do not profit from methods for optimization in continuous space)
- Points close in the latent space need to produce (for now syntactically) similar expressions; This makes exploration of 
the latent space using optimization possible.

In this section we show how to evaluate these characteristics and how to run symbolic regression experiments using HVAE.

# Reconstruction Accuracy
The code for evaluating reconstruction accuracy can be found in *src/reconstruction_accuracy.py* script. You can run this script 
from the command line with the following flags:

| Flag            | Description                                                                                                                                                                                   | Default Value |
|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| expressions     | Path to a txt file with expressions (one expression per line, symbols are separated with spaces)                                                                                              | /             |
| symbols         | Space separated symbols that can occur in expressions (currently supported symbols can be found or added in *src/symbol_library.py*), multiplication should be denoted as \\\* instead of \*. | /             |
| batch           | Batch size of the HVAE model                                                                                                                                                                  | 32            |
| num_vars        | Number of variables that appear in expressions (symbol library maps variables to upper case alphabet characters excluding C).                                                                 | 2             |
| has_const       | Add this flag if expressions also contain constants (denoted by the character C)                                                                                                              | False         |
| latent_size     | The dimension HVAE's latent space                                                                                                                                                             | 128           |
| epochs          | Number of training epochs                                                                                                                                                                     | 20            |
| annealing_iters | Number of iterations in which the $\lambda$ parameter increases                                                                                                                               | 1800          |
| verbose         | Add this flag to see the comparison between the original and predicted expressions during training                                                                                            | False         |
| seed            | Random seed of the starting fold                                                                                                                                                              | 18            |

TBA

Table below shows the percentage of syntactically correct expressions and the reconstruction accuracy (evaluated as the edit
distance between the original and the predicted expression in the postfix notation). 

![Table accuracy](https://github.com/smeznar/HVAE/blob/master/images/table_reconstruction.png)

Additionally, we show the efficiency of HVAE with regard to the number of training examples needed and the dimension of latent space below.

![efficiency](https://github.com/smeznar/HVAE/blob/master/images/efficiency.png)

# Linear interpolation
We use linear interpolation to show that points close in the latent space produce similar expressions.

TBA

![linear_interpolation](https://github.com/smeznar/HVAE/blob/master/images/li.png)

# Symbolic regression

TBA - EDHiE (Equation Discovery with Hierarchical variational autoEncoders) = HVAE + evolutionary algorithm
![symbolic_regression](https://github.com/smeznar/HVAE/blob/master/images/sr.png)

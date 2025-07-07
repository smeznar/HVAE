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

## Quickstart instructions
1. Install HVAE/EDHiE with the command: ```pip install git+https://github.com/smeznar/HVAE``` 
2. Create an instance of SRDataset using [SRToolkit](https://github.com/smeznar/SymbolicRegressionToolkit/tree/master).
3. Run the EDHiE or HVAR function.

An example can be found below:
```
from SRToolkit.dataset import SRBenchmark
``` 

## Using HVAE and EDHiE
This repository implements HVAE and EDHiE (HVAE + evolutionary algorithm), HVAR (HVAE + random sampling). HVAE is an autoencoder that needs to be trained before we are able to use it as either a generator or for equation discovery/symbolic regression.

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

**Disclaimer:** Since submission of the manuscript "Efficient generator of mathematical expressions for symbolic regression",
we changed some parts of the approach (mostly BatchedNode, regularization, and symbolic regression script) which may impact
performance.

## Reconstruction Accuracy
The code for evaluating reconstruction accuracy can be found in *EDHiE/reconstruction_accuracy.py* script together with an example of how to test it. 

Table below shows the percentage of syntactically correct expressions and the reconstruction accuracy (evaluated as the edit
distance between the original and the predicted expression in the postfix notation). 

![Table accuracy](https://github.com/smeznar/HVAE/blob/master/images/table_reconstruction.png)

Additionally, we show the efficiency of HVAE with regard to the number of training examples needed and the dimension of latent space below.

![efficiency](https://github.com/smeznar/HVAE/blob/master/images/efficiency.png)

# Linear interpolation
We use linear interpolation to show that points close in the latent space produce similar expressions. We encode expressions 
into the latent space with the encoder, generating vectors $z_A$ and $z_B$. Then we create a sequence of points $z_\alpha$
with the formula: $z_\alpha = (1-\alpha)\cdot z_A + \alpha\cdot z_B$, where $\alpha = i/n, i\in 0, ..., n$ and $n$ the 
number of points we want to create. 

To try it out use the _linear_interpolation.py_ script. Some results of linear interpolation are shown in the table below:

![linear_interpolation](https://github.com/smeznar/HVAE/blob/master/images/li.png)

## Symbolic regression
For evaluation of EDHiE (Equation Discovery with Hierarchical variational autoEncoders = HVAE + evolutionary algorithm)
on the symbolic regression task, you can use the script _symbolic_regression.py_. 

Some results of symbolic regression on the Nguyen symbolic regression benchmark can be found in the table below.
![symbolic_regression](https://github.com/smeznar/HVAE/blob/master/images/sr.png)

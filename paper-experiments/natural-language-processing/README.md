# Scaling-up DFA -- Natural Language Processing

## Organization

`train_lm.py` contains all the training logic and provides results for Table 5. `attention.py`, `radam.py`, `transformer.py`, and `utils.py` all contains code related to our implementation of the Transformer.  

## Reproducibility 

To reproduce our results: 
```bash
./run_experiments.sh
``` 

## Changes in `torchtext` since the publication of this code
As of version 0.7, `torchtext` will issue a warning about the `Field` object being deprecated and more recent dataset classes being available in `torchtext.experimental`.
If you have an even newer version of `torchtext`, our code might not run. In that case, you can either revert to an older version of `torchtext` or find the classes used in our code in `torchtext.legacy`.

We chose not to update our code to the new `torchtext` version to ensure reproducibility of the results reported in the paper.
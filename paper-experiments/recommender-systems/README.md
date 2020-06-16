# Scaling-up DFA -- Recommender Systems

## Organization

The training and testing logic are in the `recsys.py` file, used for Table 2 and A.3. `alignment.py` can be used for reproducing Table A.1. `dfa_models.py` contains the definition of all DFA models used. 


## Reproducibility
In order to reproduce the results from Table 2 in the paper 
and A.2 in the Appendix, you have to run the 
`reproduce_<model-name>.sh` scripts. 
For example to reproduce the results for `deepfm`, run

```bash
./reproduce_deepfm.sh
```

It will perform the training and print *Test AUC* and *logloss* 
for the BP, DFA and shallow configuration.

In case you have a multi-GPU machine, you can select the GPU on which 
the script is running by modifying the `GPU_ID` variable in the script.

To reproduce alignment measurement results, the Deep & Cross entry for instance:

```bash
python alignment.py --model deepcross --learning-rate 0.00005 --dropout 0. --batch-size 512
```

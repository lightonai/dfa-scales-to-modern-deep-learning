# Scaling-up DFA -- Geometric Learning

## Organization

`geometric.py` contains code to train GCNNs in general; it is the basis of Table 3 and Table A.4. `autoencoder.py` contains code for the training of graph autoencoders, it is used for Table 4. `tsne.py` is used to generate Figure 2, with the t-SNE embeddings. Finally, `alignment.py` is used to measure alignment and reproduce the results of Table A.2. `models.py` contains the description of the BP and DFA models, shared between all codes. 

## Reproducibility 

For instance, to reproduce the ChebConv entry for DFA on Cora in Table 3: 
```bash
python geometry.py --model cheb --training-method dfa --dataset-name cora
```

The other scripts work similarly, and provide full documentation for the parameters to pass. 

## A note on t-SNE

We use a CUDA implementation of t-SNE, requiring installation from source if not using `conda`. See https://github.com/CannyLab/tsne-cuda for details. 

We note this implementation is non-deterministic. 

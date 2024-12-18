# score-general-id-CRL
Code-base for the AISTATS 2024 paper: [General Identifiability and Achievability for Causal Representation Learning](https://arxiv.org/abs/2310.15450)

### **UPDATE 12/07/2024**: Please refer to the most recent version of the code at https://github.com/acarturk-e/score-based-crl

Contains the codes for reproducing results for the GSCALE-I algorithm 

Requires 'torch >= 2.0'. 

Execute "main.py" file for reproducing the simulations for the transform $X=tanh(T.Z)$ and execute "analyze.py" to generate the evaluation metrics presented in the paper.
Change $n$, $d$, ${\rm nsamples}$, ${\rm nsteps_{max}}$, and $\lambda_G$ variables appropriately for reproducing various results presented in the paper.


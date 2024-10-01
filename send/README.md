# Sensitive Neuron Dropout Expeiments

First step: pip install -r requirements.txt in the root directory of the project.

train_no_meta.py is the main file for running send on the HELM dataset given that in the MIND/auto-labeled/output dataset, your hallucination data corresponding to the model of interest is saved.

train_no_meta_medical.py is the main file for running send on the medical dataset given that in the same directory alpaca.csv file exists.

train_normal.py for the normal continual training of the model on HELM.

train_no_meta_medical.py similar but for the med dataset.

The other files in the directory contribute to the plotting code of the comparison results between 
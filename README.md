# GPTology: The Impact of Fine-Tuning on the Geometry of GPT-2.
This repo contains code for the anonymous EMNLP sumbission. The repository is broken up into 2 different scripts: ```run_models.py``` and ```analysis.py``` Code for this project should be run in the following order: 


1) ```run_models.py``` is used to fine-tune GPT-2 and generate dictionaries of hidden states on the training and validation data. 
2) ```analysis.py``` contains functions to perform Centered Kernel Alignment (CKA) analysis and to train a 1-D SVM on the principal rogue dimensions in the fine-tuned models. 

## Dependencies
```pip3 install -r requirements.txt```

## Run Models
Use ```run_models.py``` to fine-tune GPT-2 for SST-2 and QNLI. To train a model on SST-2, evaluate the perfomrance on the hidden validation data and collect hidden states on the training and validation data run: 

```python3 run_models.py --task "sst2" --train True --evaluate True --hidden_states True```

Note that setting ``--train False`` assumes that there is already a fine-trained model and will throw an error if no model has been fine-tuned. 

## Analysis
This script will perform CKA and rogue dimensions analysis. CKA analysis will output a hickle file which can be used in visualize results. 

```python3 run_models.py --task "sst2"  --rogue_dim True ```

The cka_scores() function in this script can be used to calculate the CKA scores for each model's hidden state representations. 

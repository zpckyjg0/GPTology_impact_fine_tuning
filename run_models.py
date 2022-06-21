#############################################################
#############################################################
#############################################################    
"""CODE FOR THIS FILE WAS ADAPTED FROM: 
 https://github.com/INK-USC/IsoBN
 The original code was meant for fine tuning
 BERT on GLUE tasks and looks at whether adding
 an isotropic regularizer improves performance""" 
#############################################################
#############################################################
#############################################################


#PyTorch & Numpy
import torch 
from torch.utils.data import DataLoader 
import numpy as np
#HuggingFace Stuff 
from transformers import AutoTokenizer, AutoModel
from transformers import GPT2TokenizerFast, GPT2Model, GPT2Config, GPT2LMHeadModel, GPT2ForSequenceClassification
from transformers import BertTokenizerFast, BertModel, BertConfig, BertForSequenceClassification
from transformers import DistilBertTokenizerFast, DistilBertModel, DistilBertConfig, DistilBertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset, load_metric  
import hickle
# Misc  
from tqdm import tqdm 
import argparse 
import random
import os
import pickle 
from sklearn import svm

# TODO: this collate function assumes the inputs are list-- I'm returning tensors
# Do we keep as tensors and edit this function or what? 
def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch]) 
    #max_len = max([f["input_ids"].shape[1] for f in batch]) 
    #print(max_len) 
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    outputs = { "input_ids": input_ids, "attention_mask": input_mask, "labels": labels }
    return outputs
    

#TODO Finish editing the testing flag: for now store the incorrect/correct examples
#want a way extract sentences from the batch where labels are incorrect. 
#Per element boolean if the preds and labels match 

#TODO Do we want to add a wandb wrapper??? ]
#TODO at a minimum, we need to store/print train&val loss 

def eval(args, data, model, tokenizer, test=False):
    # Set model to eval mode. Load metric and create data loader.  
    print("Evaluating") 
    model.eval() 
    eval_loader = DataLoader(data, batch_size=args.batch_size, collate_fn=collate_fn)
    
    # Send model to gpu if available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Lists to store results. 
    preds_list = []
    labels_list = []
    
    # collect correct and incorrect examples  
    correct = {} 
    correct["sentence"] = []
    correct["label"] = []
    correct["prob"] = []  
    
    incorrect = {} 
    incorrect["sentence"] = []
    incorrect["label"] = []
    incorrect["prob"] = [] 
    
    # main EVAL loop 
    for idx, batch in enumerate(eval_loader):
        # send batch to device  
        batch = {key: value.to(device) for key, value in batch.items()} 
       
        # Set model to eval and run input batches with no_grad to disable gradient calculations   
        model.eval()
        with torch.no_grad():
            outputs = model(**batch) 
            logits = outputs.logits   
       # Store Predictions and Labels
        preds = logits.argmax(axis=1)        
        preds = preds.detach().cpu().numpy()  
        preds_list.append(preds)   
        labels = batch["labels"].detach().cpu().numpy() 
        labels_list.append(labels)  
        probs = torch.nn.functional.softmax(logits, dim=1)  
         
        # Get indices of correct and incorrect labels 
        correct_indx = np.where(preds == labels)[0] 
        incorrect_indx = list(np.where(preds != labels)[0]) 
        ids = batch['input_ids'] 
         
        for i in correct_indx:
            ex = ids[i]  
            ex = ex[ex != 0] 
            correct["sentence"].append(tokenizer.decode(ex))
            correct["label"].append(labels[i])
            correct["prob"].append(max(probs[i]))
        
        for i in incorrect_indx:
            ex = ids[i]  
            ex = ex[ex != 0] 
            incorrect["sentence"].append(tokenizer.decode(ex))
            incorrect["label"].append(labels[i])
            incorrect["prob"].append(max(probs[i])) 
    # Saving results          
    with open(args.task + "_correct.pickle", "wb") as handle:  
        pickle.dump(correct, handle)   
    with open(args.task + "_incorrect.pickle", "wb") as handle:   
        pickle.dump(incorrect, handle) 
    
    # Compute Accuracy 
    preds = np.concatenate(preds_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    acc = (preds == labels).sum()/len(preds)

    return acc 

def train(args, train_data, eval_data, model):
    # Instantiate Data Loader, optimizer & scheduler 
    train_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True) 
    optimizer = args.optimizer(model.parameters(), lr=1e-5) 
    #scheduler = args.scheduler
    
    # Send model to gpu if available  
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Main TRAINING loop
    for epoch in range(args.num_epochs):
        for idx, batch in enumerate(tqdm(train_loader)):
            # Set model into training mode  
            model.train() 
            
            # Send batch to gpu if available 
            batch = {key: value.to(device) for key, value in batch.items()} 
             
            # Input batch, calculate loss and backprop 
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward() 
            #scheduler.step() 
            optimizer.step()
            
            # Clears all gradient calculations for the next round of training 
            model.zero_grad()   
        
        # After each epoch, evaluate on validation data 
        #results = eval(args, data=eval_data, model=model)         
         
        print("{} out of {} epochs completed".format(epoch+1, args.num_epochs))
        print("Loss: {}".format(loss.item())) 
        #print("Performance on Validation Set: {}".format(results)) 
        print("------------------------------------------------------------------") 
    # TODO: save model 


def get_hidden_states_qnli_gpt2(tokenizer, model, qnli_data): 
    # Create dictionary
    model.eval() 
    hidden_states = {} 
    hidden_states[12] = []  
    for j in range(len(qnli_data['train']['label'])):
        sentence = (qnli_data['train']['question'][j], qnli_data['train']['sentence'][j])
        tokens_tensor = tokenizer(*sentence, return_tensors='pt', padding=False, max_length=256, truncation=True)
        with torch.no_grad():
            outputs = model(**tokens_tensor, output_hidden_states=True)
            states = outputs.hidden_states
            hidden_states[12].append(np.squeeze(states[12])[-1])       
    return hidden_states

"""
def get_hidden_states_qnli_gpt2(tokenizer, model, qnli_data): 
    # Create dictionary
    model.eval()  
    hidden_states = {} 
    hidden_states[12] = []  
    
    # DataLoader & Device 
    data_loader = DataLoader(qnli_data, batch_size=8, collate_fn=collate_fn, shuffle=False)    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
     
    for idx, batch in enumerate(data_loader):
        batch = {key: value.to(device) for key, value in batch.items()} 
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True)
            states = outputs.hidden_states 
            hidden_states[12].append(states[12][:,-1,:])       
    
    return hidden_states
"""

def get_hidden_states_sst2_gpt2(tokenizer, model, sst2_data):
    # Create dictionary
    model.eval() 
    hidden_states = {}
    hidden_states[12] = []  
    for sentence in sst2_data: 
        tokens_tensor = tokenizer(sentence, return_tensors='pt', padding=False, max_length=256, truncation=True)
        with torch.no_grad():
            outputs = model(**tokens_tensor, output_hidden_states=True)
            states = outputs.hidden_states
            hidden_states[12].append(np.squeeze(states[12])[-1])         
    return hidden_states

def get_hidden_states_sst2_bert(tokenizer, model, sst2_data): 
    # Create dictionary
    model.eval() 
    hidden_states = {}
    for i in range(1,13):
        hidden_states[i] = []  
    for sentence in sst2_data['validation']['sentence']: 
        tokens_tensor = tokenizer(sentence, return_tensors='pt', padding=False, max_length=256, truncation=True)
        with torch.no_grad():
            outputs = model(**tokens_tensor, output_hidden_states=True)
            states = outputs.hidden_states
            for i in range(1, len(states)):
                hidden_states[i].append(np.squeeze(states[i])[0])         
    return hidden_states

def get_hidden_states_qnli_bert(tokenizer, model, qnli_data):
    """Calculates the probability negative"""
    # Create dictionary
    model.eval() 
    hidden_states = {}
    
    hidden_states[12] = []  
    for j in range(len(qnli_data['validation']['label'])):
        sentence = (qnli_data['validation']['question'][j], qnli_data['validation']['sentence'][j])
        tokens_tensor = tokenizer(*sentence, return_tensors='pt', padding=False, max_length=256, truncation=True)
        with torch.no_grad():
            outputs = model(**tokens_tensor, output_hidden_states=True)
            states = outputs.hidden_states 
            hidden_states[12].append(np.squeeze(states[12])[0])       
    return hidden_states

def get_pred_sst2(sentence, tokenizer, model):
   
    tokens_tensor = tokenizer(*sentence, return_tensors='pt', padding=False, max_length=256, truncation=True)
       
    model.eval(); 
    
    with torch.no_grad(): 
        outputs = model(**tokens_tensor)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
    return probs.argmax().item()


def get_pred_qnli(sentence, tokenizer, model):

    tokens_tensor = tokenizer(*sentence, return_tensors='pt', padding=False, max_length=256, truncation=True) 
    model.eval(); 
    
    with torch.no_grad(): 
        outputs = model(**tokens_tensor)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
    return probs.argmax().item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--task", default="sst2", type=str)
    parser.add_argument("--hidden_states", default=True, type=bool)
    parser.add_argument("--model", default="gpt2", type=str)
    parser.add_argument("--optimizer", default=AdamW)
    parser.add_argument("--seed", default=83, type=str) 
    parser.add_argument("--train", default=False, type=bool) 
    parser.add_argument("--rogue_dim", default=False, type=bool)
    #parser.add_argument("--scheduler") 
    args = parser.parse_args() 
    
    print(args) 

    # Load data 
    data = load_dataset("glue", args.task)
    num_labels = len(data["train"].features["label"].names) 
    print("Data loaded") 
    
    #Sow seeds 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
    
  
    #Instantiate the tokenizer and add the padding token 
    # TODO: does it matter what side we pad on???  
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2") 
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token 
   
    if args.model == "bert": 
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased") 
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token  
  
    if args.model == "distbert": 
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased") 
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token

    task_type = {
        "classification": [
            "sst2",
            "qnli",
        ]
    }    
    
    
    if args.task in task_type["classification"]:  
        if args.model == "gpt2": 
            model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=num_labels)
        if args.model == "bert":  
            model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=num_labels) 
        if args.model == "distbert":  
            model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-cased", num_labels=num_labels)

    #Specifying the pad token  
    model.config.pad_token_id = model.config.eos_token_id

    # TODO: add more tasks  if desired 
    task_keys = {
            "sst2": ("sentence", None),
            "qnli": ("question", "sentence"),
            } 
    
    print("Preprocessing Data")
    # preprocess data   
   
    def preprocess(example): 
        key1, key2 = task_keys[args.task] 
        if key2 is None: inputs = (example[key1],)
        else: 
            inputs = (example[key1], example[key2])
        tokenizer.pad_token = tokenizer.eos_token 
        results = tokenizer(*inputs, max_length=256, truncation=True) 
        results["labels"] = example["label"] #if "label" in example else 0  
        return results
 
     
    train_data = list(map(preprocess, data["train"])) 
    eval_data = list(map(preprocess, data["validation"])) 
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
    
    args.train = False
    
    """
    if args.train==True:   
        print("Preparing to train.")  
        train(args=args, train_data=train_data, eval_data=eval_data, model=model)
        if not os.path.isdir("../models"): os.mkdir("../models") 
        torch.save(model.state_dict(), "../models/" + args.model + "_" + args.task + ".pth") 
        print("Training complete... On to testing!")
    
    if args.train==False: 
        print("Loading model from checkpoint.")  
        model.load_state_dict(torch.load("../models/"+ args.model + "_" + args.task + ".pth", map_location=device)) 
        print("Model loaded... On to testing!")
    
    if args.hidden_states==True:  
        if args.task == "sst2": 
            print("Making hidden states for SST2")  
            #SST2 Hidden States 
            sst2_data = load_dataset("glue", "sst2") 
            sst2_train_states = get_hidden_states_sst2_gpt2(tokenizer, model, sst2_data["train"]["sentence"]) 
            hickle.dump(sst2_train_states, args.model + "_sst2_train_hidden_states.hickle", mode='w') 

            sst2_validation_states = get_hidden_states_sst2_gpt2(tokenizer, model, sst2_data["validation"]["sentence"]) 
            hickle.dump(sst2_validation_states, args.model + "_sst2_validation_hidden_states.hickle", mode='w') 
            print("hidden states saved")        
        
        if args.task == "qnli": 
            print("Making hidden states for QNLI") 
            #QNLI Hidden States 
            qnli_data = load_dataset("glue", "qnli") 
            qnli_states = get_hidden_states_qnli_gpt2(tokenizer, model,qnli_data) 
            hickle.dump(qnli_states, args.model + "_qnli_train_hidden_states.hickle", mode='w') 

            print("hidden states saved") 
     
    """    
    
    print("Loading model from checkpoint.")  
    model.load_state_dict(torch.load("../models/"+ args.model + "_" + args.task + ".pth", map_location=device)) 
    #print("Model loaded... On to testing!")
    
    if args.rogue_dim==True:    
        # load dataset labels 
        if args.task=="sst2":
            data = load_dataset("glue", "sst2") 
        else:
            data = load_dataset("glue", "qnli")
        
        # load last hidden state representations 
        print("Loading hidden states") 
        train = hickle.load(args.model + "_"+ args.task  + "_train_hidden_states.hickle") 
        train = np.vstack(train[12]) 
        val = hickle.load(args.model + "_"+ args.task  + "_validation_hidden_states.hickle") 
        val = np.vstack(val[12]) 
        # Train 1D SVM:
        print("Training SVM on dimension 496")
        clf = svm.SVC()
        clf = clf.fit(np.reshape(train[:,496], (-1,1)), data["train"]["label"])
        print("SVM trained!") 
        # Evaluate
        num_val = len(data["validation"]["label"]) 
        correct = 0  
        agree = 0  
        
        for i in range(num_val):
            clf_pred = clf.predict(np.reshape(val[i][496], (-1,1))) 
            if clf_pred == data["validation"]["label"][i]: 
                correct += 1 
            sentence = (data["validation"]["question"][i], data["validation"]["sentence"][i])
            if clf_pred == get_pred_qnli(sentence, tokenizer, model):
                agree += 1
        
        print("Performance Acc. of Dim496 on {}: {}".format(args.task, correct/num_val))
        print("Performance Agreement: {}".format(agree/num_val))  
       
       # Agreement in prediction: 

    #results = eval(args=args, data=eval_data, model=model, tokenizer=tokenizer, test=True) 
    #print("TESTING ACCURACY: ", results)
    
if __name__  == "__main__":
    main()





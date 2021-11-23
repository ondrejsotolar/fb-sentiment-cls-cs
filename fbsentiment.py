import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification, pipeline
from transformers import RobertaTokenizer, RobertaForSequenceClassification, ElectraForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import random
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
import os
from typing import List, Tuple
import json
from pathlib import Path
import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

def set_seed(seed: int):
    """
    Set reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed). Doesn't work that well.

    :param seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)


def balanced_split(target, trainSize=0.8, getTestIndexes=True, shuffle=False, seed=None):
    """
    :return Index of balanced label split. Cuts off overflow of the majority classes.
    """
    classes, counts = np.unique(target, return_counts=True)
    nPerClass = float(len(target))*float(trainSize)/float(len(classes))
    if nPerClass > np.min(counts):
        print("Insufficient data to produce a balanced training data split.")
        print("Classes found %s"%classes)
        print("Classes count %s"%counts)
        ts = float(trainSize*np.min(counts)*len(classes)) / float(len(target))
        print("trainSize is reset from %s to %s"%(trainSize, ts))
        trainSize = ts
        nPerClass = float(len(target))*float(trainSize)/float(len(classes))
    # get number of classes
    nPerClass = int(nPerClass)
    print("Data splitting on %i classes and returning %i per class"%(len(classes),nPerClass ))
    # get indexes
    trainIndexes = []
    for c in classes:
        if seed is not None:
            np.random.seed(seed)
        cIdxs = np.where(target==c)[0]
        cIdxs = np.random.choice(cIdxs, nPerClass, replace=False)
        trainIndexes.extend(cIdxs)
    # get test indexes
    testIndexes = None
    if getTestIndexes:
        testIndexes = list(set(range(len(target))) - set(trainIndexes))
    # shuffle
    if shuffle:
        trainIndexes = random.shuffle(trainIndexes)
        if testIndexes is not None:
            testIndexes = random.shuffle(testIndexes)
    # return indexes
    return trainIndexes, testIndexes


def prepare_dataset():
    """
    Prepare balanced sentiment analysis train/test splits (0.8).

    :return train_texts, train_labels, valid_texts, valid_labels
    """
    with open('gold-posts.txt', encoding='utf-8') as f:
        posts = f.readlines()
    with open('gold-labels.txt', encoding='utf-8') as f:
        labels = f.readlines()

    def to_cat(x: str) -> int:
        if x == 'p':
            return 1
        elif x == 'n':
            return 2
        else:
            return 0
    X = np.array([x.strip() for x in posts])
    y = np.array([to_cat(x.strip()) for x in labels])

    # DOES NOT WORK - too imbalanced
    #skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    #for train_index, test_index in skf.split(X, y):
    #    X_train, X_test = X[train_index], X[test_index]
    #    y_train, y_test = y[train_index], y[test_index]
    #    break
    # WORKS better
    trI, teI = balanced_split(y)

    train_texts = X[trI].tolist()
    train_labels = y[trI].tolist()
    valid_texts = X[teI].tolist()
    valid_labels = y[teI].tolist()
    return train_texts, train_labels, valid_texts, valid_labels


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    """
    :return sklearn metrics
    """
    labels = pred.label_ids
    pred = pred.predictions.argmax(-1)
    #acc = accuracy_score(labels, pred)
    #recall = recall_score(y_true=labels, y_pred=pred)
    #precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred, average='weighted')    
    return {"f1": f1}


def rmdir(directory):
    """
    Remove dir with contents.
    """
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    directory.rmdir()


def main():
    """
    Train & evaluate a Transformer model for classification.
    """
    print('# load tokenizer')
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    print('# load dataset')
    (train_texts, train_labels, valid_texts, valid_labels) = prepare_dataset()
    result = {
        'train': '{} : {} : {}'.format(len([x for x in train_labels if x == 0]), len([x for x in train_labels if x == 1]), len([x for x in train_labels if x == 2])),
        'test': '{} : {} : {}'.format(len([x for x in valid_labels if x == 0]), len([x for x in valid_labels if x == 1]), len([x for x in valid_labels if x == 2]))
    }
    print(result['train'])
    print(result['test'])
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=MAX_LENGTH)

    print('# convert tokenized data into a torch Dataset')
    train_dataset = SentimentDataset(train_encodings, train_labels)
    valid_dataset = SentimentDataset(valid_encodings, valid_labels)

    print('# load model & move to GPU')
    if MODEL_NAME == 'ufal/robeczech-base':
        model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(TARGET_NAMES)).to("cuda")
    elif MODEL_NAME == 'Seznam/small-e-czech':
        model = ElectraForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(TARGET_NAMES)).to("cuda")
    else:
        model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(TARGET_NAMES)).to("cuda")

    print('# set training params')
    w_steps = int((EPOCHS * len(train_texts)) / (3 * BATCH_DEV * VIS_DEV))
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,  # output directory
        num_train_epochs=EPOCHS,  # total number of training epochs
        per_device_train_batch_size=BATCH_DEV,  # batch size per device during training
        per_device_eval_batch_size=BATCH_DEV,  # batch size for evaluation
        warmup_steps=w_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        load_best_model_at_end=True,  # load the best model when finished training (default metric is loss)
        metric_for_best_model='f1',
        logging_steps=100,  # log & save weights each logging_steps
        evaluation_strategy="steps",  # evaluate each `logging_steps`
        learning_rate=1e-5,
        save_total_limit=3,
        disable_tqdm=True
    )
    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=valid_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # the callback that computes metrics of interest
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    print('# train')
    trainer.train()
    print('#################### eval')
    print(trainer.evaluate())

    print('#NOT  save to disk')
    #model.save_pretrained(MODEL_SAVE_PATH)
    #tokenizer.save_pretrained(MODEL_SAVE_PATH)

    def get_probs(text):
        inputs = tokenizer(text, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to("cuda")
        outputs = model(**inputs)
        return outputs[0].softmax(1)

    predictions = np.array([get_probs(valid_texts[i]).cpu().detach().numpy()[0] for i in range(len(valid_texts))])
    print('##################### F1 #########################')
    f1 = f1_score(valid_labels, np.argmax(predictions, -1), average='weighted')
    print(f1)
    cfm = confusion_matrix(valid_labels, np.argmax(predictions, -1)).tolist()
    print(cfm)

    results_pth = Path('{}.json'.format(MODEL_NAME.split('/')[1]))
    with open(results_pth, 'w', encoding='utf-8') as outfile:
        result['f1'] = f1
        result['confusion_matrix'] = cfm
        json.dump(result, outfile, ensure_ascii=False)

    rmdir(OUTPUT_DIR)
    del model
    del tokenizer
    del trainer
    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()
    print('Done: ' + str(results_pth))


if __name__ == "__main__":
    TARGET_NAMES = ['0', '1', '2']
    SEED = 1
    OUTPUT_DIR = './results'
    MAX_LENGTH = 128
    EPOCHS = 100
    BATCH_DEV = 64
    VIS_DEV = 1
    MODEL_NAME = 'Seznam/small-e-czech'
    MODEL_SAVE_PATH = '{}'.format(MODEL_NAME.split('/')[1])
    main()



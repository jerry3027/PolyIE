import torch
import config
import json
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, BertForTokenClassification
import numpy as np
from dataset import EntityDataset
from bert import Bert
import engine
from evaluation import test_fn

if __name__ == '__main__':
    # read data
    with open('./Cleaned_v1/Final_v2/train_split.txt', 'r') as train_file:
        train_dataset = json.load(train_file)
    with open('./Cleaned_v1/Final_v2/validation_split.txt', 'r') as validation_file:
        validation_dataset = json.load(validation_file)
    with open('./Cleaned_v1/Final_v2/test_split.txt', 'r') as test_file:
        test_dataset = json.load(test_file)

    train_text, train_label, train_relation = train_dataset['text'], train_dataset['label'], train_dataset['relation']
    test_text, test_label, test_relation = test_dataset['text'], test_dataset['label'], test_dataset['relation']
    validation_text, validation_label, validation_relation = validation_dataset['text'], validation_dataset['label'], validation_dataset['relation']

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = EntityDataset(train_text, train_label, train_relation, tokenizer)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE)

    validation_dataset = EntityDataset(validation_text, validation_label, validation_relation, tokenizer)
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=config.VALID_BATCH_SIZE)

    test_dataset = EntityDataset(test_text, test_label, test_relation, tokenizer)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.TRAIN_BATCH_SIZE)

    device =torch.device('cuda:7')

    model = Bert(len(config.NER_LABELS))
    # model=BertForTokenClassification.from_pretrained('allenai/scibert_scivocab_uncased', num_labels=len(config.NER_LABELS))


    model.to(device)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_dataset) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_f1 = -np.inf

    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        validation_loss = engine.eval_fn(validation_data_loader, model, device)
        print(f"Train Loss = {train_loss} Valid Loss = {validation_loss}")

        metric_dict, _, _ = test_fn(validation_data_loader, model, device)
        f1_score = metric_dict['micro avg'][2]
        print('f1_score', f1_score)
        if f1_score > best_f1:
            torch.save(model.state_dict(), config.MODEL_PATH)            
            best_f1 = f1_score

    
    metrics, _, _ = test_fn(test_data_loader, model, device)
    print('Result on test dataset')
    print(metrics)
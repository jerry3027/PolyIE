from math import ceil
import numpy as np 
import config
import torch
from tools import split_overlength_bert_input_sequence, align_labels, merge_list_of_lists


class EntityDataset():
    def __init__(self, text_list, label_list, relation_list, tokenizer, paragraph_split_length=512):
        self.tokenizer = tokenizer
        self.paragraph_split_length = paragraph_split_length

        text_list = merge_list_of_lists(text_list)
        label_list = merge_list_of_lists(label_list)
        relation_list = merge_list_of_lists(relation_list)
        
        self.text_list, self.label_list, self.relation_list = self.split_overlength_input(text_list, label_list, relation_list, self.paragraph_split_length, self.tokenizer)

    def split_overlength_input(self, text_list, label_list, relation_list, paragraph_split_length, tokenizer):
        tokenized_text_list = []
        tokenized_label_list = []
        tokenized_relation_list = []

        for i in range(len(text_list)):
            text = text_list[i]
            label = label_list[i]
            relation = relation_list[i]
                
            split_tks_seq_list = split_overlength_bert_input_sequence(text, tokenizer, paragraph_split_length, relation)
            
            tk_idx = 0
            split_label_seq_list = []
            split_relation_seq_list = []
            
            if isinstance(split_tks_seq_list[0], str):
                split_tks_seq_list = [split_tks_seq_list]
                split_label_seq_list = [label]
                split_relation_seq_list = [relation]
            else:
                for seq in split_tks_seq_list:
                    split_label_seq_list.append(label[tk_idx:tk_idx+len(seq)])
                    current_relation = []
                    for single_relation in relation:
                        if np.min(single_relation) >= tk_idx and np.max(single_relation) <= tk_idx + len(seq):
                            current_relation.append(single_relation)
                    split_relation_seq_list.append(current_relation)
                    tk_idx += len(seq)
            
            tokenized_text_list.extend(split_tks_seq_list)
            tokenized_label_list.extend(split_label_seq_list)
            tokenized_relation_list.extend(split_relation_seq_list)
            
        return tokenized_text_list, tokenized_label_list, tokenized_relation_list

    def label2id(self, labels):
        label_types = config.NER_LABELS
        label_mapping = {label: i for i, label in enumerate(label_types)}
        label_id_list = []
        for label in labels:
            if label == -100:
                label_id_list.append(-100)
            # HACK HERE
            elif label in ['B-ES', 'I-ES']:
                label_id_list.append(label_mapping['O'])
            else:
                label_id_list.append(label_mapping[label])
        return label_id_list

    def __len__(self):
        return len(self.text_list)
    
    def __getitem__(self, index):
        tokenized_text = self.tokenizer(self.text_list[index], is_split_into_words=True, padding="max_length", max_length=512)

        labels = align_labels(tokenized_text, self.label_list[index])
        ner_labels = self.label2id(labels)

        return {
            'input_ids': torch.tensor(tokenized_text['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(tokenized_text['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(tokenized_text['token_type_ids'], dtype=torch.long),
            'labels': torch.tensor(ner_labels, dtype=torch.long)
        }
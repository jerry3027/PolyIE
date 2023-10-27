import numpy as np
import itertools
import operator
from chemdataextractor.doc import Paragraph
from preprocess_dataset import convert_sentence_to_token_offset

def merge_list_of_lists(lists):
    merged = list(itertools.chain.from_iterable(lists))
    return merged


def split_overlength_bert_input_sequence(sequence, tokenizer, max_seq_length, relations=None):
    
    cdr_seq_list = convert_sentence_to_token_offset(sequence, Paragraph(' '.join(sequence)).sentences)

    tks_seq_list = []
    for seq in cdr_seq_list:
        tks_seq_list.append(sequence[seq[0]:seq[1]])

    # Merge sentences when relation crosses sentences so we don't break them
    merged_tk_seq_list = []
    if relations:
        for relation in relations:
            first_tk_idx = np.min(relation, axis=0)[0]
            last_tk_idx = np.max(relation, axis=0)[1]-1 # Because end offsets are not inclusive
            tk_len = 0
            for seq in tks_seq_list:
                if tk_len + len(seq) < first_tk_idx or (tk_len < first_tk_idx and first_tk_idx < tk_len + len(seq)):
                    merged_tk_seq_list.append(seq)
                elif first_tk_idx < tk_len and last_tk_idx > tk_len:
                    merged_tk_seq_list[-1].extend(seq)
                else:
                    merged_tk_seq_list.append(seq)
                tk_len += len(seq)
            tks_seq_list = merged_tk_seq_list
            merged_tk_seq_list = []

    tks = merge_list_of_lists(tks_seq_list)

    if len(tokenizer.tokenize(' '.join(tks), add_special_tokens=True)) < max_seq_length:
        return tks
    
    seq_bert_len_list = [len(tokenizer.tokenize(' '.join(tks_seq), add_special_tokens=True))
                         for tks_seq in tks_seq_list]

    

    if (np.asarray(seq_bert_len_list) > max_seq_length).any():
        raise ValueError("One or more sentences in the input sequence are longer than the designated maximum length.")

    split_points = [0, len(tks_seq_list)]
    split_bert_lens = [sum(seq_bert_len_list[split_points[i]:split_points[i+1]])
                       for i in range(len(split_points)-1)]

    while (np.asarray(split_bert_lens) > max_seq_length).any():
        new_split_points = list()
        for idx, bert_len in enumerate(split_bert_lens):
            if bert_len > max_seq_length:
                seq_bert_len_sub_list = seq_bert_len_list[split_points[idx]:split_points[idx+1]]
                seq_bert_len_sub_accu_list = list(itertools.accumulate(seq_bert_len_sub_list, operator.add))
                # try to separate sentences as evenly as possible
                split_offset = np.argmin((np.array(seq_bert_len_sub_accu_list) - bert_len / 2) ** 2)
                new_split_points.append(split_offset + split_points[idx] + 1)

        split_points += new_split_points
        split_points.sort()

        split_bert_lens = [sum(seq_bert_len_list[split_points[i]:split_points[i+1]])
                           for i in range(len(split_points)-1)]

    split_tks_seq_list = [merge_list_of_lists(tks_seq_list[split_points[i]:split_points[i+1]])
                          for i in range(len(split_points)-1)]

    return split_tks_seq_list

def align_labels(tokenized_text, label, label_all_tokens=False): 
    word_ids = tokenized_text.word_ids()

    previous_word_idx = None 
    aligned_labels = []

    for word_idx in word_ids: 
        if word_idx is None: 
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:              
            aligned_labels.append(label[word_idx]) 
        else: 
            aligned_labels.append(label[word_idx] if label_all_tokens else -100)
        previous_word_idx = word_idx
    return aligned_labels



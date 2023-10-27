from doctest import testfile
import json
from xmlrpc.server import SimpleXMLRPCRequestHandler
from chemdataextractor.doc import Paragraph
import numpy as np

def mmie_to_drug(dataset_path):
    with open(dataset_path, 'r') as dataset_file:
        dataset_json = json.load(dataset_file)

    drug_list = []

    for article in dataset_json:

        doc_key = article['id']
        text = article['text']
        entities = article['entities']
        relations = article['relations']
        entities = sorted(entities, key=lambda x : x['start_offset'])

        segmented_text, segmented_entities, segmented_relations = segmentation(text, entities, relations)

        align_ids(segmented_text, segmented_entities, segmented_relations)

        for i in range(len(segmented_text)):
            drug_entry = {}
            drug_entry['doc_id'] = str(doc_key) + '_' + str(i)
            drug_entry['sentence'] = segmented_text[i]
            drug_entry['spans'] = segmented_entities[i]
            drug_entry['rels'] = segmented_relations[i]
            drug_entry['paragraph'] = None
            drug_entry['source'] = None
            drug_list.append(drug_entry)

    return drug_list

def align_ids(segmented_text, segmented_entities, segmented_relations):
    for i in range(len(segmented_text)):
        entities = segmented_entities[i]
        old_id2new_id = {}
        for j in range(len(entities)):
            old_id2new_id[entities[j]['span_id']] = j
            entities[j]['span_id'] = j
        for rel in segmented_relations[i]:
            new_spans = []
            for item in rel['spans']:
                new_spans.append(old_id2new_id[item])
            rel['spans'] = new_spans

def segmentation(texts, entities, relations):
    segmented_text = []
    segmented_entities = []
    segmented_relations = []
    
    # Obtain sentece breakdown
    text = ' '.join(texts)
    sentences = Paragraph(text).sentences
    sentence_breakdown, sentence_char_breakdown = convert_sentence_to_offset(texts, sentences)
    sentence_breakdown.sort(key=lambda x : x[0])
    sentence_char_breakdown.sort(key=lambda x : x[0])

    sentence_start2sentence_char_start = {}
    for i in range(len(sentence_breakdown)):
        sentence_start2sentence_char_start[sentence_breakdown[i][0]] = sentence_char_breakdown[i][0]
    
    # Obtain relation breakdown
    entity_id2entity_content = {mention['id'] : mention for mention in entities}
    relation2relation_bd = {}
    for relation in relations:
        min_idx = float('inf')
        max_idx = float('-inf')
        for item in relation:
            if entity_id2entity_content[item]['start_offset'] < min_idx:
                min_idx = entity_id2entity_content[item]['start_offset']
            if entity_id2entity_content[item]['end_offset'] > max_idx:
                max_idx = entity_id2entity_content[item]['end_offset']
        relation2relation_bd[json.dumps(relation)] = [min_idx, max_idx]
    # relation_breakdown = merge_intervals(relation_breakdown)

    # Find min len sentence that contains the relation
    sen_bd2rel = {}
    for relation in relation2relation_bd:
        min_idx = relation2relation_bd[relation][0]
        max_idx = relation2relation_bd[relation][1]

        start = 0
        end = 0
        # Greatest start less than min_idx 
        for sen_bd in sentence_breakdown:
            if sen_bd[0] < min_idx:
                start = sen_bd[0]
        # Least finish grater than max_idx
        for sen_bd in reversed(sentence_breakdown):
            if sen_bd[1] > max_idx:
                end = sen_bd[1] 

        if (start, end) not in sen_bd2rel:
            sen_bd2rel[(start,end)] = [json.loads(relation)]
        else:
            sen_bd2rel[(start,end)].append(json.loads(relation))

    sen_bd2rel = merge_sentence_bd(sen_bd2rel)

    for key in sen_bd2rel:
        segmented_text.append(' '.join(texts[key[0]:key[1]]))

        label_segmentation = []
        for entity in entities:
            if entity['start_offset'] >= key[0] and entity['end_offset'] <= key[1]:
                label_segmentation.append({"span_id": entity['id'], 
                                            "text": ' '.join(texts[entity['start_offset']:entity['end_offset']]),
                                            "start": entity['char_start_offset']-sentence_start2sentence_char_start[key[0]],
                                            "end": entity['char_end_offset']-sentence_start2sentence_char_start[key[0]],
                                            "token_start": entity['start_offset']-key[0], 
                                            "token_end": entity['end_offset']-key[0],
                                            "type": entity['label']})
        segmented_entities.append(label_segmentation)

        relation_segmentation = []
        for rel in sen_bd2rel[key]:
            relation_segmentation.append({"class":"COMB", "spans":rel, "is_context_needed": False})
        segmented_relations.append(relation_segmentation)

    return segmented_text, segmented_entities, segmented_relations
            

def merge_sentence_bd(sen_bd2rel):
    intervals = sorted(sen_bd2rel.keys(), key= lambda x : x[0])
    merged_sen_bd = []
    rels = []
    for interval in intervals:
        # if the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged_sen_bd or merged_sen_bd[-1][1] < interval[0]:
            merged_sen_bd.append(list(interval))
            rels.append(sen_bd2rel[interval])
        else:
        # otherwise, there is overlap, so we merge the current and previous
        # intervals.
            merged_sen_bd[-1][1] = max(merged_sen_bd[-1][1], interval[1])
            rels[-1].extend(sen_bd2rel[interval])
    result = {}
    for i in range(len(merged_sen_bd)):
        result[tuple(merged_sen_bd[i])] = rels[i]
    return result

# Convert character offset to token offset
def convert_sentence_to_offset(tokenized_text, sentences):
    sentence_token_offset = []
    sentence_char_offset = []

    token_idx = 0
    character_idx = 0
    sentence_idx = 0
    while token_idx < len(tokenized_text) and sentence_idx < len(sentences):
        current_sentence = sentences[sentence_idx]
        # add start
        if not sentence_token_offset or len(sentence_token_offset[-1]) == 2:
            if current_sentence.start <= character_idx:
                sentence_token_offset.append([token_idx])
                sentence_char_offset.append([character_idx])
        else:
        # add end
            if current_sentence.end <= character_idx:
                sentence_token_offset[-1].append(token_idx)
                sentence_char_offset[-1].append(character_idx-1)
                # add next start
                sentence_token_offset.append([token_idx])
                sentence_char_offset.append([character_idx])
                sentence_idx += 1
        character_idx += len(tokenized_text[token_idx]) + 1
        token_idx += 1
    
    # Add to last sentence
    sentence_token_offset[-1].append(token_idx)
    sentence_char_offset[-1].append(character_idx-1)

    return sentence_token_offset, sentence_char_offset


if __name__ == '__main__':
    train_path = './Cleaned_v1/Final/train_split.txt'
    validation_path = './Cleaned_v1/Final/validation_split.txt'
    test_path = './Cleaned_v1/Final/test_split.txt'

    train_pure_list = mmie_to_drug(train_path)
    validation_pure_list = mmie_to_drug(validation_path)
    test_pure_list = mmie_to_drug(test_path)

    train_output_path = './Baselines/drug-combo-extraction/mmie_data/train.txt'
    with open(train_output_path, 'w+') as train_file:
        for item in train_pure_list:
            train_file.write(json.dumps(item))
            train_file.write('\n')

    with open(train_output_path, 'a') as validation_file:
        for item in validation_pure_list:
            validation_file.write(json.dumps(item))
            validation_file.write('\n')

    test_output_path = './Baselines/drug-combo-extraction/mmie_data/test.txt'
    with open(test_output_path, 'w+') as test_file:
        for item in test_pure_list:
            test_file.write(json.dumps(item))
            test_file.write('\n')



import json
from chemdataextractor.doc import Paragraph

def read_dataset(dataset_path):
    with open(dataset_path, 'r') as dataset_file:
        dataset_json = json.load(dataset_file)
    tokenized_text = []
    labels = []

    all_entities = []
    all_relations = []

    for article in dataset_json:
        text = article['text']
        entities = article['entities']
        relations = article['relations']
        entities = sorted(entities, key=lambda x : x['start_offset'])

        tokenized_text.append(text)
        current_labels = []
        
        token_idx = 0
        entity_idx = 0
        while token_idx < len(text):
            if entity_idx >= len(entities):
                current_labels.append('O')
            elif token_idx < entities[entity_idx]['start_offset']:
                current_labels.append('O')
            elif token_idx == entities[entity_idx]['start_offset']:
                current_labels.append('B-' + entities[entity_idx]['label'])
            elif token_idx > entities[entity_idx]['start_offset'] and token_idx < entities[entity_idx]['end_offset']:
                current_labels.append('I-' + entities[entity_idx]['label'])
            elif token_idx == entities[entity_idx]['end_offset']:
                entity_idx += 1
                if entity_idx < len(entities) and token_idx == entities[entity_idx]['start_offset']:
                    current_labels.append('B-' + entities[entity_idx]['label'])
                else:
                    current_labels.append('O')
            else:
                current_labels.append('O')
            token_idx += 1
        
        labels.append(current_labels)

        all_entities.append(entities)
        all_relations.append(relations)
    
    all_text, all_label, all_relation = segmentation(tokenized_text, labels, all_entities, all_relations)

    return all_text, all_label, all_relation

# Still need to add relation
def segmentation(texts, labels, all_entities, all_relations):
    all_text = []
    all_label = []
    all_relation = []

    for i, tokenized_text in enumerate(texts):
        text = ' '.join(tokenized_text)
        entities = all_entities[i]
        entities = {mention['id'] : mention for mention in entities}
        label = labels[i]
        relations = all_relations[i]

        current_text = []
        current_label = []
        current_relation = []

        # Obtain sentece breakdown
        sentences = Paragraph(text).sentences
                
        sentence_breakdown = convert_sentence_to_token_offset(tokenized_text, sentences)
        
        sentence_breakdown.sort(key=lambda x : x[0])

        # Obtain relation breakdown
        relation_breakdown_to_relations = {} # Keep track of relation mapping
        relation_breakdown = []
        for relation in relations:
            min_idx = float('inf')
            max_idx = float('-inf')
            new_relation = []
            for item in relation:
                if entities[item]['start_offset'] < min_idx:
                    min_idx = entities[item]['start_offset']
                if entities[item]['end_offset'] > max_idx:
                    max_idx = entities[item]['end_offset']
                new_relation.append([entities[item]['start_offset'], entities[item]['end_offset']])
            relation_breakdown.append([min_idx, max_idx])
            relation_breakdown_to_relations[(min_idx, max_idx)] = new_relation
        
        relation_breakdown.sort(key=lambda x : x[0])
        relation_breakdown = merge_intervals(relation_breakdown)

        # Segment based on 3 rules: 1) less than 512, 2) does not break down relation, 3) does not break down sentence
        sentence_idx = 0
        relation_idx = 0

        tk_idx = 0

        while sentence_idx < len(sentence_breakdown):
            start = sentence_breakdown[sentence_idx][0]
            end = sentence_breakdown[sentence_idx][1]
            while sentence_idx + 1 < len(sentence_breakdown) and sentence_breakdown[sentence_idx+1][1] - start < 512:
                sentence_idx += 1
                end = sentence_breakdown[sentence_idx][1]
                
                while relation_idx < len(relation_breakdown) and relation_breakdown[relation_idx][1] < end:
                    relation_idx += 1
            # we are breaking relation
            while relation_idx < len(relation_breakdown) and relation_breakdown[relation_idx][1] >= end and relation_breakdown[relation_idx][0] < end:
                sentence_idx -= 1
                end = sentence_breakdown[sentence_idx][1]
                
                while relation_breakdown[relation_idx][0] > end:
                    relation_idx -= 1
                
            # add to segmentation
            current_text.append(tokenized_text[start:end])
            current_label.append(label[start:end])

            relation_segmentation = []

            for key in relation_breakdown_to_relations:
                if key[0] >= start and key[1] <= end:
                    for item in relation_breakdown_to_relations[key]:
                        item[0] -= tk_idx
                        item[1] -= tk_idx
                    relation_segmentation.append(relation_breakdown_to_relations[key])
            
            current_relation.append(relation_segmentation)
            
            tk_idx += end - start
            sentence_idx += 1
    
        all_text.append(current_text)
        all_label.append(current_label)
        all_relation.append(current_relation)

    return all_text, all_label, all_relation
            
# Convert character offset to token offset
def convert_sentence_to_token_offset(tokenized_text, sentences):
    sentence_token_offset = []

    token_idx = 0
    character_idx = 0
    sentence_idx = 0
    while token_idx < len(tokenized_text) and sentence_idx < len(sentences):
        current_sentence = sentences[sentence_idx]
        # add start
        if not sentence_token_offset or len(sentence_token_offset[-1]) == 2:
            if current_sentence.start <= character_idx:
                sentence_token_offset.append([token_idx])
        else:
        # add end
            if current_sentence.end <= character_idx:
                sentence_token_offset[-1].append(token_idx)
                # add next start
                sentence_token_offset.append([token_idx])
                sentence_idx += 1
        character_idx += len(tokenized_text[token_idx]) + 1
        token_idx += 1
    
    # Add to last sentence
    sentence_token_offset[-1].append(token_idx)

    return sentence_token_offset

def merge_intervals(intervals):
    merged = []
    for interval in intervals:
        # if the list of merged intervals is empty or if the current
        # interval does not overlap with the previous, simply append it.
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
        # otherwise, there is overlap, so we merge the current and previous
        # intervals.
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged
        
if __name__ =='__main__':
    text_train, label_train, relation_train = read_dataset('./Cleaned_v1/Final/train_split.txt')
    with open('./Cleaned_v1/Final_v2/train_split.txt', 'w+') as train_file:
        train_content = {}
        train_content['text'] = text_train
        train_content['label'] = label_train
        train_content['relation'] = relation_train
        json.dump(train_content, train_file)
    text_test, label_test, relation_test = read_dataset('./Cleaned_v1/Final/test_split.txt')
    with open('./Cleaned_v1/Final_v2/test_split.txt', 'w+') as test_file:
        test_content = {}
        test_content['text'] = text_test
        test_content['label'] = label_test
        test_content['relation'] = relation_test
        json.dump(test_content, test_file)
    text_validation, label_validation, relation_validation = read_dataset('./Cleaned_v1/Final/validation_split.txt')
    with open('./Cleaned_v1/Final_v2/validation_split.txt', 'w+') as validation_file:
        validation_content = {}
        validation_content['text'] = text_validation
        validation_content['label'] = label_validation
        validation_content['relation'] = relation_validation
        json.dump(validation_content, validation_file)
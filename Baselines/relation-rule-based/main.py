from xml.dom.minidom import DocumentType
import jsonlines

def read_dataset(dataset_path, negative_rel_path):
    data_raw = list(jsonlines.open(dataset_path))
    negative_rel_raw = list(jsonlines.open(negative_rel_path))
    doc_id2negative_rel_raw = {negative_rel['doc_id']:negative_rel for negative_rel in negative_rel_raw}
    
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for document_raw in data_raw:
        doc_id = document_raw['doc_id']
        sentence = document_raw['sentence']
        spans = sorted(document_raw['spans'], key=lambda x : x['start'])
        rels = document_raw['rels']

        negative_rels = doc_id2negative_rel_raw[doc_id]['negative_samples']
        
        span_id2start = {span['span_id']:span['start'] for span in spans}

        for rel in rels:
            
            rel_span = rel['spans']

            cn_start = 0
            pn_start = 0 
            pv_start = 0

            match_flag = True

            for i in range(len(rel_span)):
                # We are looking at CN
                if i == 0:
                    cn_start = span_id2start[rel_span[i]]
                # We are looking at PN
                elif i == 1:
                    closest_id = None
                    for span in spans:
                        if span['type'] == 'PN':
                            if closest_id == None:
                                closest_id = span['span_id']
                            elif abs(cn_start - span_id2start[closest_id]) > abs(cn_start - span['start']):
                                closest_id = span['span_id']
                    if closest_id == None or closest_id != rel_span[1]:
                        match_flag = False
                        break
                    pn_start = span_id2start[closest_id]
                # We are looking at PV
                elif i == 2:
                    closest_id = None
                    for span in spans:
                        if span['type'] == 'PV':
                            if closest_id == None:
                                closest_id = span['span_id']
                            elif abs(pn_start - span_id2start[closest_id]) > abs(pn_start - span['start']):
                                closest_id = span['span_id']
                    if closest_id == None or closest_id != rel_span[2]:
                        match_flag = False
                        break
                    pv_start = span_id2start[closest_id]                
                # We are looking at Condition 
                elif i == 3:
                    closest_id = None
                    for span in spans:
                        if span['type'] == 'Condition':
                            if closest_id == None:
                                closest_id = span['span_id']
                            elif abs(pv_start - span_id2start[closest_id]) > abs(pv_start - span['start']):
                                closest_id = span['span_id']
                    if closest_id == None or closest_id != rel_span[3]:
                        match_flag = False
                        break
        
            if match_flag:
                tp += 1
            else:
                fn += 1
            
        for rel in negative_rels:
            
            rel_span = rel['spans']

            cn_start = 0
            pn_start = 0 
            pv_start = 0

            match_flag = True

            for i in range(len(rel_span)):
                # We are looking at CN
                if i == 0:
                    cn_start = span_id2start[rel_span[i]]
                # We are looking at PN
                elif i == 1:
                    closest_id = None
                    for span in spans:
                        if span['type'] == 'PN':
                            if closest_id == None:
                                closest_id = span['span_id']
                            elif abs(cn_start - span_id2start[closest_id]) > abs(cn_start - span['start']):
                                closest_id = span['span_id']
                    if closest_id == None or closest_id != rel_span[1]:
                        match_flag = False
                        break
                    pn_start = span_id2start[closest_id]
                # We are looking at PV
                elif i == 2:
                    closest_id = None
                    for span in spans:
                        if span['type'] == 'PV':
                            if closest_id == None:
                                closest_id = span['span_id']
                            elif abs(pn_start - span_id2start[closest_id]) > abs(pn_start - span['start']):
                                closest_id = span['span_id']
                    if closest_id == None or closest_id != rel_span[2]:
                        match_flag = False
                        break
                    pv_start = span_id2start[closest_id]                
                # We are looking at Condition 
                elif i == 3:
                    closest_id = None
                    for span in spans:
                        if span['type'] == 'Condition':
                            if closest_id == None:
                                closest_id = span['span_id']
                            elif abs(pv_start - span_id2start[closest_id]) > abs(pv_start - span['start']):
                                closest_id = span['span_id']
                    if closest_id == None or closest_id != rel_span[3]:
                        match_flag = False
                        break
        
            if match_flag:
                fp += 1
            else:
                tn += 1
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


if __name__ == '__main__':
    print(read_dataset('./Baselines/drug-combo-extraction/mmie_data/test.txt', './Baselines/drug-combo-extraction/mmie_data/negative_relations/test.txt'))
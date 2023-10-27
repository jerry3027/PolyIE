import config
from tqdm import tqdm
import torch
from torch.functional import F
import json
from bert import Bert
from dataset import EntityDataset
from seqeval import metrics
from seqeval.scheme import IOB2
from transformers import AutoTokenizer

def id2label(id):
    label_types = config.NER_LABELS
    label_mapping = {i: label for i, label in enumerate(label_types)}
    if id == -100:
        return 'O'
    if id in ['B-ES', 'I-ES']:
        return 'O'
    return label_mapping[id]

def test_fn(data_loader, model, device):
    model.eval()
    
    true_lbs = []
    pred_lbs = []

    # For TSNE
    embeddings = None
    labels = None

    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)

        with torch.no_grad():
            _, logits, batch_embeddings, batch_labels = model(**data)
        
        logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
        logits = logits.detach().cpu().numpy()

        
        label_ids = data['labels'].to('cpu').numpy()
        mask = data['attention_mask'].to('cpu').numpy()

        for i, ids in enumerate(label_ids):
            pred_lbs.append([id2label(item) for idx, item in enumerate(logits[i]) if mask[i][idx] and ids[idx] != -100])
            true_lbs.append([id2label(item) for idx, item in enumerate(ids) if mask[i][idx] and ids[idx] != -100])

        # For TSNE
        if embeddings == None and labels == None:
            embeddings = batch_embeddings
            labels = batch_labels
        else:
            embeddings = torch.cat((embeddings, batch_embeddings), 0)
            labels = torch.cat((labels, batch_labels), 0)

    metric_dict = dict()
    report = metrics.classification_report(
        true_lbs, pred_lbs, output_dict=True, zero_division=0
    )
    for tp, results in report.items():
        metric_dict[tp] = [results['precision'], results['recall'], results['f1-score']]
    return metric_dict, embeddings, labels


if __name__ == '__main__':
    with open('./Cleaned_v1/Final_v2/train_split.txt', 'r') as test_file:
        test_dataset = json.load(test_file)

    test_text, test_label, test_relation = test_dataset['text'], test_dataset['label'], test_dataset['relation']

    tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert')

    test_dataset = EntityDataset(test_text, test_label, test_relation, tokenizer=tokenizer)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.TRAIN_BATCH_SIZE)

    device =torch.device('cuda:7')

    model = Bert(len(config.NER_LABELS))
    model.to(device)
    
    model.load_state_dict(torch.load('./Baselines/Bert_NER/Trained_models/matSciBert_model.bin'))

    result, embeddings, labels = test_fn(test_data_loader, model, device)
    
    torch.save(embeddings, './Baselines/Tsne/embeddings/matSciBert_embedding.pt')
    torch.save(labels, './Baselines/Tsne/embeddings/matSciBert_train_labels.pt')

    print(result)
# Usage
# python test_only.py --checkpoint-path checkpoints/ --test-file data/dev_set_error_analysis.jsonl \
#                     --outputs-directory /tmp/outputs/ --error-analysis-file /tmp/error_analysis.csv
import os
from unittest import result
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse
import json
import jsonlines
import pytorch_lightning as pl
import json
import torch
import sys
sys.path.append('.')
sys.path.append('..')
from common.constants import DOUBLE, ENTITY_END_MARKER, ENTITY_START_MARKER, CN_ONLY, PN_ONLY, PV_ONLY, CN_AND_PN, CN_AND_PV, PN_AND_PV
from common.utils import construct_row_id_idx_mapping, set_seed, write_error_analysis_file, write_jsonl, adjust_data, filter_overloaded_predictions
from modeling.model import RelationExtractor, load_model
from preprocessing.data_loader import  DrugSynergyDataModule
from preprocessing.preprocess import create_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-path', type=str, required=False, default="checkpoints", help="Path to pretrained Huggingface Transformers model")
parser.add_argument('--test-file', type=str, required=False, default="data/dev_set_error_analysis.jsonl")
parser.add_argument('--batch-size', type=int, default=100   , help="Batch size for testing (larger batch -> faster evaluation)")
parser.add_argument('--outputs-directory', type=str, required=False, help="Output directory where we write predictions, for offline evaluation", default="/tmp/outputs/.tsv")
parser.add_argument('--error-analysis-file', type=str, required=False, help="Output file containing error analysis information", default="test_output.tsv")
parser.add_argument('--seed', type=int, required=False, default=2021)
parser.add_argument('--avg_embedding', required=False, action='store_true')

if __name__ == "__main__":
    args = parser.parse_args()
    set_seed(args.seed)
    model, tokenizer, metadata = load_model(args.checkpoint_path)
    if ENTITY_START_MARKER not in tokenizer.vocab:
        tokenizer.add_tokens([ENTITY_START_MARKER])
    if ENTITY_END_MARKER not in tokenizer.vocab:
        tokenizer.add_tokens([ENTITY_END_MARKER])
    model.eval()

    test_data_raw = list(jsonlines.open(args.test_file))
    # TODO(Vijay): add `add_no_combination_relations`, `only_include_binary_no_comb_relations`, `include_paragraph_context`,
    # `context_window_size` to the model's metadata

    negative_relation_generation_mode=None

    test_data = create_dataset(test_data_raw,
                               label2idx=metadata.label2idx,
                               negative_relation_generation_mode=negative_relation_generation_mode,
                               add_no_combination_relations=True,
                               only_include_binary_no_comb_relations=metadata.only_include_binary_no_comb_relations,
                               include_paragraph_context=metadata.include_paragraph_context,
                               context_window_size=metadata.context_window_size,
                               produce_all_subsets=False,
                               save_file_name=None,
                               negative_samples_file='test.txt',
                               negative_sample_count=99999)
    row_id_idx_mapping, idx_row_id_mapping = construct_row_id_idx_mapping(test_data)
    
    dm = DrugSynergyDataModule(None,
                               test_data,
                               tokenizer,
                               metadata.label2idx,
                               row_id_idx_mapping,
                               train_batch_size=args.batch_size,
                               dev_batch_size=args.batch_size,
                               test_batch_size=args.batch_size,
                               max_seq_length=metadata.max_seq_length,
                               balance_training_batch_labels=False,
                               avg_embedding=args.avg_embedding)
    dm.setup()

    system = RelationExtractor(model, 0, tokenizer=tokenizer)

    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        resume_from_checkpoint=os.path.join(args.checkpoint_path, "model.chkpt")
    )
    trainer.test(system, datamodule=dm)

    # embeddings = None 
    # labels = None
    # for i, relation_embeddings in enumerate(model.relation_embeddings):
    #     relation_labels = model.relation_labels[i]
    #     if embeddings == None and labels == None:
    #         embeddings = relation_embeddings
    #         labels = relation_labels
    #     else:
    #         embeddings = torch.cat((embeddings, relation_embeddings), 0)
    #         labels = torch.cat((labels, relation_labels), 0)

    # torch.save(embeddings, './embeddings/relation_embedding.pt')
    # torch.save(labels, './embeddings/relation_labels.pt')
    
    test_predictions = system.test_predictions

    test_row_ids = [idx_row_id_mapping[row_idx] for row_idx in system.test_row_idxs]
    
    tp = 0
    fp = 0 
    fn = 0 
    tn = 0


    for i in range(len(test_predictions)):
        cur_prediction = test_predictions[i]
        cur_gold = json.loads(test_row_ids[i])['relation_label']
        # print("cur_gold", cur_gold)
        if cur_prediction == 1 and cur_gold == 1:
            tp += 1
        elif cur_prediction == 1 and cur_gold == 0:
            fp += 1
        elif cur_prediction == 0 and cur_gold == 1:
            fn += 1
        elif cur_prediction == 0 and cur_gold == 0:
            tn += 1

    print(tp)
    print(fp)
    print(fn)
    print(tn)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / len(test_predictions)
    f1 = 2 * precision * recall / (precision + recall)
    
    with open(os.path.join(args.checkpoint_path, 'results.txt'), 'w+') as result_file:
        result_file.write('precision: ' + str(precision))
        result_file.write('\n')
        result_file.write('recall: ' + str(recall))
        result_file.write('\n')
        result_file.write('accuracy: ' + str(accuracy))
        result_file.write('\n')
        result_file.write('f1: ' + str(f1))

    print("precision", precision)
    print("recall", recall)
    print("accuracy", accuracy)
    print("f1", f1)


    
    # fixed_test = filter_overloaded_predictions(adjust_data(test_row_ids, test_predictions))
    # os.makedirs(args.outputs_directory, exist_ok=True)
    # test_output = os.path.join(args.outputs_directory, "predictions.jsonl")

    # write_jsonl(fixed_test, test_output)
    write_error_analysis_file(test_data, test_data_raw, test_row_ids, test_predictions, args.error_analysis_file)
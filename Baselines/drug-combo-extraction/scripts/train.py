'''
Usage
python train.py --pretrained-lm allenai/scibert_scivocab_uncased --num-train-epochs 10 --lr 2e-4 \
--batch-size 18 --context-window-size 300 --max-seq-length 512 --label2idx data/label2idx.json \
--model-name baseline_model
'''
import os
from tokenize import Double
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse
import json
import jsonlines
import os
import pytorch_lightning as pl
from transformers import AutoTokenizer
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

import sys
sys.path.extend(["..", "."])
from common.constants import ENTITY_END_MARKER, ENTITY_START_MARKER, NOT_COMB, CN_ONLY, PN_ONLY, PV_ONLY, CN_AND_PN, CN_AND_PV, PN_AND_PV, SINGLE, DOUBLE
from preprocessing.data_loader import DrugSynergyDataModule
from preprocessing.preprocess import create_dataset
from modeling.model import BertForRelation, RelationExtractor
from common.utils import construct_row_id_idx_mapping, ModelMetadata, save_metadata, set_seed, write_error_analysis_file, write_jsonl, adjust_data, filter_overloaded_predictions

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained-lm', type=str, required=False, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", help="Path to pretrained Huggingface Transformers model")
parser.add_argument('--training-file', type=str, required=False, default="data/train_set.jsonl")
parser.add_argument('--test-file', type=str, required=False, default="data/test_set.jsonl")
parser.add_argument('--label2idx', type=str, required=False, default="data/label2idx.json")
parser.add_argument('--batch-size', type=int, required=False, default=18) # This number is good for training on an 11GB Tesla K80 GPU.
parser.add_argument('--dev-train-split', type=float, required=False, default=0.1, help="Fraction of the training set to hold out for validation")
parser.add_argument('--max-seq-length', type=int, required=False, default=512, help="Maximum subword length of the document passed to the encoder, including inserted marker tokens")
parser.add_argument('--preserve-case', action='store_true')
parser.add_argument('--num-train-epochs', default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument('--label-sampling-ratios', default=None, type=str, help="Loss weights (json list) for up/downsampling training examples of each class for training (due to label imbalance)")
parser.add_argument('--label-loss-weights', default=None, type=str, help="Loss weights (json list) for negative class labels in training (to help with label imbalance)")
parser.add_argument('--ignore-no-comb-relations', action='store_true', help="If true, then don't mine NO_COMB negative relations from the relation annotations.")
parser.add_argument('--only-include-binary-no-comb-relations', action='store_true', help="If true, and we are including no-comb relations, then only mine binary no-comb relations (ignoring n-ary no-comb relations)")
parser.add_argument('--ignore-paragraph-context', action='store_true', help="If true, only look at each entity-bearing sentence and ignore its surrounding context.")
parser.add_argument('--lr', default=2e-4, type=float, help="Learning rate")
parser.add_argument('--unfreezing-strategy', type=str, choices=["all", "final-bert-layer", "BitFit"], default="BitFit", help="Whether to finetune all bert layers, just the final layer, or bias terms only.")
parser.add_argument('--context-window-size', type=int, required=False, default=300, help="Amount of cross-sentence context to use (including the sentence in question")
parser.add_argument('--balance-training-batch-labels', action='store_true', help="If true, load training batches to ensure that each batch contains samples of each class.")
parser.add_argument('--model-name', type=str, required=True)
parser.add_argument('--seed', type=int, required=False, default=2000)
parser.add_argument('--avg_embedding', required=False, action='store_true')
parser.add_argument('--negative_generation_mode', required=False, type=str, default=None, help="One of CN, PN, PV, CNPN, CNPV, PNPV, SINGLE, DOUBLE")
parser.add_argument('--save_file_name', required=False, type=str, default=None, help="File to save negative generated samples")
parser.add_argument('--negative_samples_file', required=False, type=str, default=None, help="File to read negative generated samples")
parser.add_argument('--negative_sample_count', required=False, type=int, default=None, help="Number of negative sample to correct sample")
parser.add_argument('--training_documents', required=False, type=float, default=None, help="Number of samples to use for training")



if __name__ == "__main__":
    args = parser.parse_args()

    set_seed(args.seed)

    training_data_raw = list(jsonlines.open(args.training_file))
    test_data_raw = list(jsonlines.open(args.test_file))
    label2idx = json.load(open(args.label2idx))
    label2idx[NOT_COMB] = 0

    if args.label_sampling_ratios is None:
        label_sampling_ratios = [1.0 for _ in set(label2idx.values())]
    else:
        label_sampling_ratios = json.loads(args.label_sampling_ratios)

    if args.label_loss_weights is None:
        label_loss_weights = [1.0 for _ in set(label2idx.values())]
    else:
        label_loss_weights = json.loads(args.label_loss_weights)
    

    include_paragraph_context = not args.ignore_paragraph_context
    training_data = create_dataset(training_data_raw,
                                   label2idx=label2idx,
                                   negative_relation_generation_mode=args.negative_generation_mode,
                                   label_sampling_ratios=label_sampling_ratios,
                                   add_no_combination_relations=not args.ignore_no_comb_relations,
                                   only_include_binary_no_comb_relations=args.only_include_binary_no_comb_relations,
                                   include_paragraph_context=include_paragraph_context,
                                   context_window_size=args.context_window_size,
                                   save_file_name=args.save_file_name,
                                   negative_samples_file=args.negative_samples_file,
                                   negative_sample_count=args.negative_sample_count,
                                   training_documents=args.training_documents,
                                   seed=args.seed)
    label_values = sorted(set(label2idx.values()))
    num_labels = len(label_values)
    assert label_values == list(range(num_labels))
    assert len(label_sampling_ratios) == num_labels
    assert len(label_loss_weights) == num_labels
    test_data = create_dataset(test_data_raw,
                               label2idx=label2idx,
                               negative_relation_generation_mode=None,
                               add_no_combination_relations=not args.ignore_no_comb_relations,
                               only_include_binary_no_comb_relations=args.only_include_binary_no_comb_relations,
                               include_paragraph_context=include_paragraph_context,
                               context_window_size=args.context_window_size)
    row_id_idx_mapping, idx_row_id_mapping = construct_row_id_idx_mapping(training_data + test_data)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_lm, do_lower_case=not args.preserve_case)
    tokenizer.add_tokens([ENTITY_START_MARKER, ENTITY_END_MARKER])

    dm = DrugSynergyDataModule(training_data,
                               test_data,
                               tokenizer,
                               label2idx,
                               row_id_idx_mapping,
                               train_batch_size=args.batch_size,
                               dev_batch_size=args.batch_size,
                               test_batch_size=args.batch_size,
                               dev_train_ratio=args.dev_train_split,
                               max_seq_length=args.max_seq_length,
                               balance_training_batch_labels=args.balance_training_batch_labels,
                               avg_embedding=args.avg_embedding)
    dm.setup()

    model = BertForRelation.from_pretrained(
            args.pretrained_lm,
            cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE),
            num_rel_labels=num_labels,
            max_seq_length=args.max_seq_length,
            unfreeze_all_bert_layers=args.unfreezing_strategy=="all",
            unfreeze_final_bert_layer=args.unfreezing_strategy=="final-bert-layer",
            unfreeze_bias_terms_only=args.unfreezing_strategy=="BitFit")

    # Add rows to embedding matrix if not large enough to accomodate special tokens.
    if len(tokenizer) > len(model.bert.embeddings.word_embeddings.weight):
        model.bert.resize_token_embeddings(len(tokenizer))

    num_train_optimization_steps = len(dm.train_dataloader()) * float(args.num_train_epochs)

    if set(label_loss_weights) != {1.0}:
        # Unless all labels are being weighted equally, then compute specific label weights for class-weighted loss.
        label_loss_weighting = [w / sum(label_loss_weights) for w in label_loss_weights]
    else:
        label_loss_weighting = None

    system = RelationExtractor(model, num_train_optimization_steps, lr=args.lr, tokenizer=tokenizer, label_weights=label_loss_weighting)
    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        max_epochs=args.num_train_epochs,
    )
    trainer.fit(system, datamodule=dm)
    model_dir = "checkpoints_" + args.model_name
    model.save_pretrained(model_dir)
    trainer.save_checkpoint(os.path.join(model_dir, "model.chkpt"))
    tokenizer.save_pretrained(os.path.join(model_dir, "tokenizer"))
    metadata = ModelMetadata(args.pretrained_lm,
                             args.max_seq_length,
                             num_labels,
                             label2idx,
                             not args.ignore_no_comb_relations,
                             args.only_include_binary_no_comb_relations,
                             include_paragraph_context,
                             args.context_window_size)
    save_metadata(metadata, model_dir)
    trainer.test(system, datamodule=dm)
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
    
    


    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / len(test_predictions)
    f1 = 2 * precision * recall / (precision + recall)
    
    print("precision", precision)
    print("recall", recall)
    print("accuracy", accuracy)
    print("f1", f1)

    # fixed_test = filter_overloaded_predictions(adjust_data(test_row_ids, test_predictions))
    # os.makedirs("outputs", exist_ok=True)
    # test_output = os.path.join("outputs", args.model_name + "_predictions.jsonl")
    # write_jsonl(fixed_test, test_output)
    # write_error_analysis_file(test_data, test_data_raw, test_row_ids, test_predictions, os.path.join("outputs", args.model_name + ".tsv"))

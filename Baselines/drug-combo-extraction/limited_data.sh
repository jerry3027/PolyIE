#!/usr/bin/env bash

python scripts/train.py --model-name few_shot_0.2 \
                --pretrained-lm allenai/scibert_scivocab_uncased \
                --num-train-epochs 10 \
                --lr 2e-4 \
                --batch-size 8 \
                --training-file mmie_data/train.txt \
                --test-file mmie_data/test.txt \
                --context-window-size 400 \
                --max-seq-length 512 \
                --label2idx mmie_data/label2idx.json \
                --seed 2022 \
                --ignore-paragraph-context \
                --unfreezing-strategy final-bert-layer \
                --negative_samples_file ./all_20.txt \
                --negative_sample_count 20 \
                --training_documents 0.2

python scripts/train.py --model-name few_shot_0.4 \
                --pretrained-lm allenai/scibert_scivocab_uncased \
                --num-train-epochs 10 \
                --lr 2e-4 \
                --batch-size 8 \
                --training-file mmie_data/train.txt \
                --test-file mmie_data/test.txt \
                --context-window-size 400 \
                --max-seq-length 512 \
                --label2idx mmie_data/label2idx.json \
                --seed 2022 \
                --ignore-paragraph-context \
                --unfreezing-strategy final-bert-layer \
                --negative_samples_file ./all_20.txt \
                --negative_sample_count 20 \
                --training_documents 0.4

python scripts/train.py --model-name few_shot_0.6 \
                --pretrained-lm allenai/scibert_scivocab_uncased \
                --num-train-epochs 10 \
                --lr 2e-4 \
                --batch-size 8 \
                --training-file mmie_data/train.txt \
                --test-file mmie_data/test.txt \
                --context-window-size 400 \
                --max-seq-length 512 \
                --label2idx mmie_data/label2idx.json \
                --seed 2022 \
                --ignore-paragraph-context \
                --unfreezing-strategy final-bert-layer \
                --negative_samples_file ./all_20.txt \
                --negative_sample_count 20 \
                --training_documents 0.6

python scripts/train.py --model-name few_shot_0.8 \
                --pretrained-lm allenai/scibert_scivocab_uncased \
                --num-train-epochs 10 \
                --lr 2e-4 \
                --batch-size 8 \
                --training-file mmie_data/train.txt \
                --test-file mmie_data/test.txt \
                --context-window-size 400 \
                --max-seq-length 512 \
                --label2idx mmie_data/label2idx.json \
                --seed 7 \
                --ignore-paragraph-context \
                --unfreezing-strategy final-bert-layer \
                --negative_samples_file ./all_20.txt \
                --negative_sample_count 20 \
                --training_documents 0.8

python scripts/train.py --model-name few_shot_1.0_new \
                --pretrained-lm allenai/scibert_scivocab_uncased \
                --num-train-epochs 10 \
                --lr 2e-4 \
                --batch-size 8 \
                --training-file mmie_data/train.txt \
                --test-file mmie_data/test.txt \
                --context-window-size 400 \
                --max-seq-length 512 \
                --label2idx mmie_data/label2idx.json \
                --seed 50 \
                --ignore-paragraph-context \
                --unfreezing-strategy final-bert-layer \
                --negative_samples_file ./all_20.txt \
                --negative_sample_count 20 \
                --training_documents 99999
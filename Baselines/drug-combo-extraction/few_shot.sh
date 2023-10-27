#!/usr/bin/env bash

python scripts/train.py --model-name few_shot_5_1 \
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
                --training_documents 5

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_few_shot_5_1 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --seed 7

python scripts/train.py --model-name few_shot_5_2 \
                --pretrained-lm allenai/scibert_scivocab_uncased \
                --num-train-epochs 10 \
                --lr 2e-4 \
                --batch-size 8 \
                --training-file mmie_data/train.txt \
                --test-file mmie_data/test.txt \
                --context-window-size 400 \
                --max-seq-length 512 \
                --label2idx mmie_data/label2idx.json \
                --seed 77 \
                --ignore-paragraph-context \
                --unfreezing-strategy final-bert-layer \
                --negative_samples_file ./all_20.txt \
                --training_documents 5

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_few_shot_5_2 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --seed 77

python scripts/train.py --model-name few_shot_5_3 \
                --pretrained-lm allenai/scibert_scivocab_uncased \
                --num-train-epochs 10 \
                --lr 2e-4 \
                --batch-size 8 \
                --training-file mmie_data/train.txt \
                --test-file mmie_data/test.txt \
                --context-window-size 400 \
                --max-seq-length 512 \
                --label2idx mmie_data/label2idx.json \
                --seed 777 \
                --ignore-paragraph-context \
                --unfreezing-strategy final-bert-layer \
                --negative_samples_file ./all_20.txt \
                --training_documents 5

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_few_shot_5_3 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --seed 777










python scripts/train.py --model-name few_shot_10_1 \
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
                --training_documents 10

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_few_shot_10_1 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --seed 7

python scripts/train.py --model-name few_shot_10_2 \
                --pretrained-lm allenai/scibert_scivocab_uncased \
                --num-train-epochs 10 \
                --lr 2e-4 \
                --batch-size 8 \
                --training-file mmie_data/train.txt \
                --test-file mmie_data/test.txt \
                --context-window-size 400 \
                --max-seq-length 512 \
                --label2idx mmie_data/label2idx.json \
                --seed 77 \
                --ignore-paragraph-context \
                --unfreezing-strategy final-bert-layer \
                --negative_samples_file ./all_20.txt \
                --training_documents 10

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_few_shot_10_2 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --seed 77

python scripts/train.py --model-name few_shot_10_3 \
                --pretrained-lm allenai/scibert_scivocab_uncased \
                --num-train-epochs 10 \
                --lr 2e-4 \
                --batch-size 8 \
                --training-file mmie_data/train.txt \
                --test-file mmie_data/test.txt \
                --context-window-size 400 \
                --max-seq-length 512 \
                --label2idx mmie_data/label2idx.json \
                --seed 777 \
                --ignore-paragraph-context \
                --unfreezing-strategy final-bert-layer \
                --negative_samples_file ./all_20.txt \
                --training_documents 10

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_few_shot_10_3 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --seed 777










python scripts/train.py --model-name few_shot_20_1 \
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
                --training_documents 20

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_few_shot_20_1 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --seed 7

python scripts/train.py --model-name few_shot_20_2 \
                --pretrained-lm allenai/scibert_scivocab_uncased \
                --num-train-epochs 10 \
                --lr 2e-4 \
                --batch-size 8 \
                --training-file mmie_data/train.txt \
                --test-file mmie_data/test.txt \
                --context-window-size 400 \
                --max-seq-length 512 \
                --label2idx mmie_data/label2idx.json \
                --seed 77 \
                --ignore-paragraph-context \
                --unfreezing-strategy final-bert-layer \
                --negative_samples_file ./all_20.txt \
                --training_documents 20

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_few_shot_20_2 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --seed 77

python scripts/train.py --model-name few_shot_20_3 \
                --pretrained-lm allenai/scibert_scivocab_uncased \
                --num-train-epochs 10 \
                --lr 2e-4 \
                --batch-size 8 \
                --training-file mmie_data/train.txt \
                --test-file mmie_data/test.txt \
                --context-window-size 400 \
                --max-seq-length 512 \
                --label2idx mmie_data/label2idx.json \
                --seed 777 \
                --ignore-paragraph-context \
                --unfreezing-strategy final-bert-layer \
                --negative_samples_file ./all_20.txt \
                --training_documents 20

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_few_shot_20_3 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --seed 777










python scripts/train.py --model-name few_shot_50_1 \
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
                --training_documents 50

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_few_shot_50_1 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --seed 7

python scripts/train.py --model-name few_shot_50_2 \
                --pretrained-lm allenai/scibert_scivocab_uncased \
                --num-train-epochs 10 \
                --lr 2e-4 \
                --batch-size 8 \
                --training-file mmie_data/train.txt \
                --test-file mmie_data/test.txt \
                --context-window-size 400 \
                --max-seq-length 512 \
                --label2idx mmie_data/label2idx.json \
                --seed 77 \
                --ignore-paragraph-context \
                --unfreezing-strategy final-bert-layer \
                --negative_samples_file ./all_20.txt \
                --training_documents 50

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_few_shot_50_2 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --seed 77

python scripts/train.py --model-name few_shot_50_3 \
                --pretrained-lm allenai/scibert_scivocab_uncased \
                --num-train-epochs 10 \
                --lr 2e-4 \
                --batch-size 8 \
                --training-file mmie_data/train.txt \
                --test-file mmie_data/test.txt \
                --context-window-size 400 \
                --max-seq-length 512 \
                --label2idx mmie_data/label2idx.json \
                --seed 777 \
                --ignore-paragraph-context \
                --unfreezing-strategy final-bert-layer \
                --negative_samples_file ./all_20.txt \
                --training_documents 50

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_few_shot_50_3 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --seed 777










python scripts/train.py --model-name few_shot_100_1 \
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
                --training_documents 100

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_few_shot_100_1 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --seed 7

python scripts/train.py --model-name few_shot_100_2 \
                --pretrained-lm allenai/scibert_scivocab_uncased \
                --num-train-epochs 10 \
                --lr 2e-4 \
                --batch-size 8 \
                --training-file mmie_data/train.txt \
                --test-file mmie_data/test.txt \
                --context-window-size 400 \
                --max-seq-length 512 \
                --label2idx mmie_data/label2idx.json \
                --seed 77 \
                --ignore-paragraph-context \
                --unfreezing-strategy final-bert-layer \
                --negative_samples_file ./all_20.txt \
                --training_documents 100

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_few_shot_100_2 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --seed 77

python scripts/train.py --model-name few_shot_100_3 \
                --pretrained-lm allenai/scibert_scivocab_uncased \
                --num-train-epochs 10 \
                --lr 2e-4 \
                --batch-size 8 \
                --training-file mmie_data/train.txt \
                --test-file mmie_data/test.txt \
                --context-window-size 400 \
                --max-seq-length 512 \
                --label2idx mmie_data/label2idx.json \
                --seed 777 \
                --ignore-paragraph-context \
                --unfreezing-strategy final-bert-layer \
                --negative_samples_file ./all_20.txt \
                --training_documents 100

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_few_shot_100_3 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --seed 777
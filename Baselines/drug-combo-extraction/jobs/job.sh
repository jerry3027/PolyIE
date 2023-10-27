python scripts/train.py --model-name single_5 \
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
                --save_file_name ./single_5.txt \
                --negative_sample_count 5 \
                --negative_generation_mode SINGLE

python scripts/train.py --model-name single_10 \
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
                --save_file_name ./single_10.txt \
                --negative_sample_count 10 \
                --negative_generation_mode SINGLE

python scripts/train.py --model-name single_20 \
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
                --save_file_name ./single_20.txt \
                --negative_sample_count 20 \
                --negative_generation_mode SINGLE
    
python scripts/train.py --model-name double_5 \
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
                --save_file_name ./double_5.txt \
                --negative_sample_count 5 \
                --negative_generation_mode DOUBLE

python scripts/train.py --model-name double_10 \
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
                --save_file_name ./double_10.txt \
                --negative_sample_count 10 \
                --negative_generation_mode DOUBLE

python scripts/train.py --model-name double_20 \
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
                --save_file_name ./double_20.txt \
                --negative_sample_count 20 \
                --negative_generation_mode DOUBLE

python scripts/train.py --model-name all_5 \
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
                --save_file_name ./all_5.txt \
                --negative_sample_count 5

python scripts/train.py --model-name all_10 \
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
                --save_file_name ./all_10.txt \
                --negative_sample_count 10

python scripts/train.py --model-name all_20 \
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
                --save_file_name ./all_20.txt \
                --negative_sample_count 20

# MatsciBert
python scripts/train.py --model-name all_20_matsci \
                --pretrained-lm m3rg-iitd/matscibert \
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
                --negative_sample_count 20

python scripts/train.py --model-name double_20_matsci \
                --pretrained-lm m3rg-iitd/matscibert \
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
                --negative_samples_file ./double_20.txt \
                --negative_sample_count 20 \
                --negative_generation_mode DOUBLE

python scripts/train.py --model-name single_20_matsci \
                --pretrained-lm m3rg-iitd/matscibert \
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
                --negative_samples_file ./single_20.txt \
                --negative_sample_count 20 \
                --negative_generation_mode SINGLE

# Average Embedding
python scripts/train.py --model-name avg_double_20_matsci \
                --pretrained-lm m3rg-iitd/matscibert \
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
                --negative_samples_file ./double_20.txt \
                --negative_sample_count 20 \
                --negative_generation_mode DOUBLE \
                --avg_embedding
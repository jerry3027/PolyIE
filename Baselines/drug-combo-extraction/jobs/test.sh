python scripts/test_only.py \
    --checkpoint-path ./checkpoints_single_5 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --outputs-directory checkpoints_cn_curruption_model/outputs/ \
    --seed 2022

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_single_10 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --outputs-directory checkpoints_cn_curruption_model/outputs/ \
    --seed 2022

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_single_20 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --outputs-directory checkpoints_cn_curruption_model/outputs/ \
    --seed 2022

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_double_5 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --outputs-directory checkpoints_cn_curruption_model/outputs/ \
    --seed 2022

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_double_10 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --outputs-directory checkpoints_cn_curruption_model/outputs/ \
    --seed 2022

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_double_20 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --outputs-directory checkpoints_cn_curruption_model/outputs/ \
    --seed 2022

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_all_5 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --outputs-directory checkpoints_cn_curruption_model/outputs/ \
    --seed 2022

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_all_10 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --outputs-directory checkpoints_cn_curruption_model/outputs/ \
    --seed 2022

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_all_20 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --outputs-directory checkpoints_cn_curruption_model/outputs/ \
    --seed 2022
# MatSciBert
python scripts/test_only.py \
    --checkpoint-path ./checkpoints_all_20_matsci \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --outputs-directory checkpoints_cn_curruption_model/outputs/ \
    --seed 2022

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_single_20_matsci \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --outputs-directory checkpoints_cn_curruption_model/outputs/ \
    --seed 2022

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_double_20_matsci \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --outputs-directory checkpoints_cn_curruption_model/outputs/ \
    --seed 2022

# Average Embedding
python scripts/test_only.py \
    --checkpoint-path ./checkpoints_avg_double_20_matsci \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --outputs-directory checkpoints_cn_curruption_model/outputs/ \
    --seed 2022


# Few shot 
python scripts/test_only.py \
    --checkpoint-path ./checkpoints_few_shot_5 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --outputs-directory checkpoints_cn_curruption_model/outputs/ \
    --seed 2022

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_few_shot_10 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --outputs-directory checkpoints_cn_curruption_model/outputs/ \
    --seed 2022

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_few_shot_20 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --outputs-directory checkpoints_cn_curruption_model/outputs/ \
    --seed 2022

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_few_shot_50 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --outputs-directory checkpoints_cn_curruption_model/outputs/ \
    --seed 100

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_few_shot_100 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --outputs-directory checkpoints_cn_curruption_model/outputs/ \
    --seed 2000


# Limited Data
python scripts/test_only.py \
    --checkpoint-path ./checkpoints_few_shot_0.2 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --outputs-directory checkpoints_cn_curruption_model/outputs/ \
    --seed 2022

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_few_shot_0.4 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --outputs-directory checkpoints_cn_curruption_model/outputs/ \
    --seed 2022

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_few_shot_0.6 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --outputs-directory checkpoints_cn_curruption_model/outputs/ \
    --seed 2022

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_few_shot_0.8 \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --outputs-directory checkpoints_cn_curruption_model/outputs/ \
    --seed 2022

python scripts/test_only.py \
    --checkpoint-path ./checkpoints_few_shot_1.0_new \
    --test-file ./mmie_data/test.txt \
    --batch-size 100 \
    --outputs-directory checkpoints_cn_curruption_model/outputs/ \
    --seed 50
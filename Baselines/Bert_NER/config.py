import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 5
MODEL_PATH = './Baselines/Bert_NER/Trained_models/base_Bert_model.bin'
NER_LABELS = ['O', 'B-CN', 'I-CN', 'B-PN', 'I-PN', 'B-PV', 'I-PV', 'B-Condition', 'I-Condition']
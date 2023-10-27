import json


if __name__ == '__main__':
    # read data
    with open('./Cleaned_data/Final_v2/train_split.txt', 'r') as train_file:
        train_dataset = json.load(train_file)
    with open('./Cleaned_data/Final_v2/validation_split.txt', 'r') as validation_file:
        validation_dataset = json.load(validation_file)
    with open('./Cleaned_data/Final_v2/test_split.txt', 'r') as test_file:
        test_dataset = json.load(test_file)

    print(" ".join(train_dataset['text'][1][4]))
    print(" ".join(train_dataset['label'][1][4]))
# This is a sample Python script.
from doctest import testfile
from fileinput import filename
import numpy as np
import os
import re
import json
from sklearn.model_selection import train_test_split
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

fk_matrixes = dict()
article_list_file = "all_2.txt"
with open(article_list_file, 'r') as dataset_jsonl:
    article_list = list(dataset_jsonl)

article_list = article_list[0]
article_list = json.loads(article_list)
for article in article_list:

    id = article["id"]
    text = article['text']
    words = list(text)
    fk_matrix = np.zeros((len(words), 5))
    fk_matrixes[id] = fk_matrix

label_match = {"CN": 0, "PN": 1, "PV": 2, "Condition": 3}

def build_matrix(dataset_path):
    tokenized_dataset = []
    with open(dataset_path, 'r') as dataset_jsonl:
        article_list = list(dataset_jsonl)
    for article in article_list:
        article = json.loads(article)
        article = article[0]

        id = article['id']
        entities = article['entities']
        entities = sorted(entities, key=lambda x: x['start_offset'])

        for entity in entities:
            start_offset = entity["start_offset"]
            end_offset = entity["end_offset"]
            label_idx = label_match[entity["label"]]

            for idx in range(start_offset, end_offset + 1):
                fk_matrixes[id][idx, label_idx] += 1

def calculate_fk_value():
    all_matrix = None
    for matrix in fk_matrixes.values():
        if all_matrix is None:
            all_matrix = matrix
        else:
            all_matrix = np.concatenate((all_matrix, matrix), axis=0)

    # all_matrix = all_matrix[~np.all(all_matrix == 0, axis=1)]
    for row_idx in range(all_matrix.shape[0]):
        labeled = np.sum(all_matrix[row_idx])
        if labeled != 3:
            all_matrix[row_idx][4] = 3 - labeled

    json_matrix = {"json_matrix": all_matrix.tolist()}
    with open('fk_after_clean5.txt', 'w+') as all_file:
        json.dump(json_matrix, all_file)

    print(all_matrix.shape)
    N, _ = all_matrix.shape
    n = 3

    # calculate Pe
    cols_sum = np.sum(all_matrix, axis=0)
    print(["columns_sum", cols_sum])
    all_sum = np.sum(cols_sum)
    print(["all_sum", all_sum])
    print((cols_sum/all_sum)**2)
    Pe = np.sum((cols_sum/all_sum)**2)
    print(["Pe", Pe])

    # calculate P0
    const = 1/(N*n*(n-1))
    rest = np.sum(all_matrix**2) - N*n
    P0 = const * rest

    print(P0, Pe)
    fk_value = (P0-Pe) / (1-Pe)
    print(fk_value)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    raters_files = ["all_1.txt", "all_2.txt", "all_3.txt"]

    for rater_file in raters_files:
        build_matrix(rater_file)

    calculate_fk_value()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/

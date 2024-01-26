# PolyIE

Code and Data for arXiv paper: [PolyIE: A Dataset of Information Extraction from Polymer Material Scientific Literature](https://arxiv.org/abs/2311.07715).

## Dependencies
This repo is built with python 3.8. Run the following commands to install dependencies:
```shell
pip install -r ./requirements.txt
pip install git+https://github.com/titipata/scipdf_parser
python -m spacy download en_core_web_sm
pip install PyMuPDF
pip install decimer-segmentation
pip install tensorflow
pip install BeautifulSoup4
cde data download
```
## Dataset
The Annotation/Data/Ner folder contains manually annotated articles on the Polymer Solar Cells dataset and the Lithium Batteries dataset. The text in these articles are parsed from PDFs, pre-annotated with noisy labels, and manually annotated in Doccano.

The annotated labels contains Compound Name (CN), Property Names (PN), Property Values (PV), and Conditions (Conditions). To view the annotation in Doccano, create a new dataset and upload the jsonl file to Doccano.

The Annotation/Data/Relation folder contains manually annotated articles with relations. This is completed based on the annotated Ner files. The relation annotation connects related entities to form <CN, PN, PV, Condition> an n-ary tuple.

Running the extraction file on PDFs:


## Running the parse pipeline to generate noisy labels
To run the parse pipeline, you will need to specify the input directory, output directory, and a mentrion list. Example command is provided below

Run the GROBID using the given bash script before parsing PDF:

bash serve_grobid.sh

Then run the following command to start the parse pipeline: 

python main_pipeline.py --pdf_folder ./Data/PDFs --output_folder ./Data/Output --mention_dict power conversion efficiencies

--pdf_folder specifies the folder that contains PDF files

--output_folder specifices the folder to output the parsed results 

--mention_dict specifies the keywords that are used to match property names

The parse pipeline will parse text from the PDF files, extract chemical name mentions, property name mentions, and property value mentions. In addition, it will also extract all molecular images from the PDF files.

## Running baselines
Install the following dependencies in order to run the baselines:
```
pip install transformers
pip install torch
pip install numpy 
```

To run Bert based NER baselines, use the following command
```
python ./Baselines/Bert_NER/main.py
```

To run dygiepp, PURE, and drug-combo-extract baselines, navigate to the corresponding folder and follow the instructions in the readme files. 

To run GPT related baselines, install the following dependencies:
```
pip install openai
```
To run GPT based NER baselines, use the following command:
```
python ./GPT/baseline_gpt_ner.py
```

To run GPT based RE baselines, use the following command:
```
python ./GPT/baseline_gpt_re.py
```

## Experiment results

<img width="579" alt="Screen Shot 2023-01-17 at 1 17 19 PM" src="https://user-images.githubusercontent.com/62039540/213014236-a77d0b16-a567-4777-b215-521265acb10a.png">

The table above shows some NER baselines on our manually curated dataset.



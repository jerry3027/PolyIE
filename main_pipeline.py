from lib2to3.pgen2.tokenize import tokenize
import os
import json
import matplotlib.pyplot as plt
from sympy import diophantine
import time
import scipdf
import fitz
import re
import cv2
from PIL import Image
import io
import shutil
from decimer_segmentation import segment_chemical_structures, segment_chemical_structures_from_file
# from DECIMER import predict_SMILES
from DatabaseAPIs.pubChemAPI import pubChem_REST_Smile, pubChem_REST_id2synonyms
from chemdataextractor.doc import Document, Paragraph
from chemdataextractor.nlp.tokenize import ChemWordTokenizer

PSC_MENTION_DICT = ['power', 'conversion', 'efficiency', 'PCE', 'fill', 'factor', 'open', 'circuit', 'voltage', 'hole', 'mobility', 'band', 'gap', 'absorption', 'edge', 'HOMO', 'LUMO', 'short', 'current', 'VOC', 'JSC', 'FF', 'electron', 'EG', 'highest', 'occupied', 'lowest', 'unoccupied', 'molecular', 'orbital']
KOLON_MENTION_DICT = ['ionic', 'conductivity', 'electrochemical', 'stability', 'window', 'tensile', 'strength', "Young's", 'modulus', 'elongation', 'peak', 'capacity', 'retention', 'IC', 'ESW', 'conductivity', 'discharge', 'temperature']
YINGHAO_MENTION_DICT = ['ultimate', 'enthalpy', 'polymerization', 'temperature', 'Tcr', 'equilibrium', 'concentration', 'ceiling', 'temperature', 'temperatures', 'Tc', 'crystallinity', 'degree', 'melting', 'range']
UNIT_DICT = ['nm', 'C', '%', 'cm', 'V', 's', 'eV', 'Å', '°C', 'mA', 'mW', 'AM']


# Parser object that automatically parse text and images from pdfs
class Parser():
    def __init__(self, pdf_path, output_path, save_sections=True, extract_images=True, extract_molecular_image=True):
        # Path to paper pdf
        self.pdf_path = pdf_path
        # Output folder that contains the parsed text and images
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        self.sections = self.extractText(pdf_path)
        if save_sections:
            with open(os.path.join(output_path, 'sections.txt'), 'w+') as file:
                file.write(json.dumps(self.sections))

        if extract_images:
            self.extractImage(pdf_path, output_path)

        if extract_molecular_image:
            self.extractMolecularImage(figures_path=os.path.join(output_path,'figures'), output_path=output_path)


    # Parse paper to obtain all sections' text
    def extractText(self, pdf_path):
        article_dict = scipdf.parse_pdf_to_dict(pdf_path)
        sections = []
        sections.append({'heading':'ABSTRACT', 'text':article_dict['abstract']})
        for section_dict in article_dict['sections']:
            heading = section_dict['heading']
            heading = re.sub(r'[^a-zA-Z ]', '', heading)
            text = section_dict['text']
            sections.append({'heading': heading, 'text': text})
        return sections

    # Parse paper to obtain all images
    def extractImage(self, pdf_path, output_path):
        doc = fitz.open(pdf_path)
        image_output_path = os.path.join(output_path, 'figures')
        if not os.path.exists(image_output_path):
            # shutil.rmtree(image_output_path)
            os.mkdir(image_output_path)
        for page_idx, page in enumerate(doc):
            images = page.get_images()
            if images:
                for idx, image in enumerate(images):
                    xref = image[0]
                    base_img = doc.extract_image(xref)
                    img_bytes = base_img['image']
                    img_ext = base_img['ext']
                    Image.open(io.BytesIO(img_bytes)).save(open(os.path.join(image_output_path, f"image{page_idx+1}_{idx+1}.{img_ext}"), 'wb'))

    def extractMolecularImage(self, figures_path, output_path):
        figures = os.listdir(figures_path)
        figures.sort()
        if figures:
            for figure in figures:
                fig = cv2.imread(os.path.join(figures_path, figure))
                print(figure)
                segments = segment_chemical_structures(fig, expand=True)
                image_output_path = os.path.join(output_path, 'molecular structure images')
                if not os.path.exists(image_output_path):
                        os.mkdir(image_output_path)
                for i in range(len(segments)):
                    plt.imshow(segments[i], interpolation='nearest')
                    plt.savefig(os.path.join(image_output_path, figure + str(i) + '.png'))

def tokenize_passage(paper_path, output_path):
    cwt = ChemWordTokenizer()
    with open(paper_path, 'r') as file:
        article_dict = json.load(file)
    tokenized_sections = []
    sentences = []
    for paragrah_dict in article_dict:
        heading = paragrah_dict['heading']
        section = paragrah_dict['text']

        if heading not in [" EXPERIMENTAL SECTION", "EXPERIMENTAL SECTION", " EXPERIMENTAL PART", "", "Author Contributions", " ACKNOWLEDGMENTS", " ASSOCIATED CONTENT", " S Supporting Information", "Experimental section Measurements and characterization", "Notes", "Experimental", "Acknowledgements", "Notes and references", "Experimental Materials and synthesis", "Experimental Section", " EXPERIMENTAL METHODS", "Experimental"]:
            tokenized_section = cwt.tokenize(section)
            tokenized_sections.append(tokenized_section)
            para = Paragraph(section)
            paragraph_sentences = []
            for sentence in para.sentences:
                paragraph_sentences.append((sentence.start, sentence.end))
            sentences.append(paragraph_sentences)
    with open(output_path, 'w+') as file:
            output_dict = {'tokenized_sections': tokenized_sections, 'sentences': sentences}
            json.dump(output_dict, file)
    return tokenized_sections


def extractMentions(paper_path, type, mention_dict, whole_text=True):
    with open(paper_path, 'r') as file:
        output_dict = json.load(file)
        tokenized_paragraphs = output_dict['tokenized_sections']
        sentences = output_dict['sentences']
    mentions = []
    if not whole_text:
        abstract = tokenized_paragraphs[0]
        abstract_sentences = sentences[0]
        paragraph_mentions = processParagraph(abstract, abstract_sentences, type, mention_dict)
        mentions.append(paragraph_mentions)
    else:
        # Iterate through each section of article
        for tokenized_paragraph_idx, tokenized_paragraph in enumerate(tokenized_paragraphs):
            paragraph_mentions = processParagraph(tokenized_paragraph, sentences[tokenized_paragraph_idx], type, mention_dict)
            mentions.append(paragraph_mentions)
    return mentions

def processParagraph(tokenized_paragraph, sentences, type, mention_dict):
    character_offset = 0
    paragraph_mentions = []
    if type == 'ChemDataExtractor':
        paragraph_string = ' '.join(tokenized_paragraph)
        doc = Document(Paragraph(paragraph_string))
        cems = doc.cems
        for i, token in enumerate(tokenized_paragraph):
            for cem in cems: 
                if character_offset == cem.start:
                    paragraph_mentions.append((i, i+1, token, 'CN'))
                elif paragraph_mentions and paragraph_mentions[len(paragraph_mentions)-1][2] + ' ' + token in cem.text:
                    old_mention = paragraph_mentions[len(paragraph_mentions)-1]
                    mention = (old_mention[0], old_mention[1]+1, old_mention[2] + ' ' + token, old_mention[3])
                    paragraph_mentions[len(paragraph_mentions)-1] = mention
            character_offset += len(token) + 1
    if type == 'Rules':
        sentence_idx = 0
        sentences_of_interest = []
        for i, token in enumerate(tokenized_paragraph):
            if token in mention_dict:
                # Taking care of spans for token length longer than 1
                if paragraph_mentions and paragraph_mentions[len(paragraph_mentions)-1][1] == i:
                    old_mention = paragraph_mentions[len(paragraph_mentions)-1]
                    mention = (old_mention[0], old_mention[1]+1, old_mention[2] + ' ' + token, old_mention[3])
                    paragraph_mentions[len(paragraph_mentions)-1] = mention
                else:
                    paragraph_mentions.append((i, i+1, token, 'PN'))
                # Marking sentence of interest
                while sentence_idx < len(sentences) and character_offset > sentences[sentence_idx][1]:
                    sentence_idx += 1
                # Adding space for each character might result in sentence_idx > len(sentences)
                if sentence_idx < len(sentences):
                    sentences_of_interest.append(sentences[sentence_idx])
            character_offset += len(token) + 1
        # Second pass search for numerical tokens
        character_offset = 0
        for i, token in enumerate(tokenized_paragraph):
            if tools.isNumeric(token):
                for sentence_of_interest in sentences_of_interest:
                    if character_offset >= sentence_of_interest[0] and character_offset < sentence_of_interest[1]:
                        if paragraph_mentions and paragraph_mentions[len(paragraph_mentions)-1][1] == i:
                            old_mention = paragraph_mentions[len(paragraph_mentions)-1]
                            mention = (old_mention[0], old_mention[1]+1, old_mention[2] + ' ' + token, old_mention[3])
                            paragraph_mentions[len(paragraph_mentions)-1] = mention
                            break
                        else:
                            paragraph_mentions.append((i, i+1, token, 'PV'))
                            break
            character_offset += len(token) + 1
    if type == 'PubChem':
        pass
    return paragraph_mentions

# One token can be mapped to multiple chemical mentions
# Does not enforce sentence limitations
def formRelations(mentions):
    cn = []
    pn = []
    pv = []
    relations = []
    for mention in mentions:
        if mention[3] == 'CN':
            cn.append(mention)
        elif mention[3] == 'PN':
            pn.append(mention)
        elif mention[3] == 'PV':
            pv.append(mention)
    for chemical_name in cn:
        pn = sorted(pn, key=lambda x: relationComparator(x, chemical_name))
        pv = sorted(pv, key=lambda x: relationComparator(x, chemical_name))
        # relation = [chemical_name, '', pn[0] if pn else '', pv[0] if pv else '']
        relation = [chemical_name[2], '', pn[0][2] if pn else '', pv[0][2] if pv else '']
        relations.append(relation)
    return relations

def relationComparator(x, chemical_tuple):
    if x[0] > chemical_tuple[1]:
        return x[0] - chemical_tuple[1]
    elif x[0] < chemical_tuple[0]:
        return chemical_tuple[0] - x[0]
    else:
        return 0


if __name__ == '__main__':
    # For PSC
    # pdf_path = './Data/PolymerSolarCell/PDFs'
    # output_path = './Outputs/PSC'

    # For Kolon
    # downloadPSC(doi_path='./Data/Kolon/kolon_dois.txt', output_path='./Data/Kolon/PDFs')
    # pdf_path = './Data/Kolon/PDFs'
    # output_path = './Outputs/Kolon'

    # For Yinghao
    # doi_list = obtain_Yinghao_doi(dataset_path='./Data/Datasets/Yinghao/jsonl', output_file='./Data/Datasets/Yinghao/yinghao_doi.txt')
    # downloadPSC(doi_path='./Data/Datasets/Yinghao/yinghao_doi.txt', output_path='./Data/Datasets/Yinghao/PDFs')
    # pdf_path = './Data/Datasets/Yinghao/PDFs'
    # output_path = './Outputs/Yinghao'

    # For Pranav
    # doi_list = obtain_Pranav_doi(dataset_path='./Data/Datasets/Pranav/parsed_files', output_file='./Data/Datasets/Pranav/pranav_doi.txt')
    # downloadPSC(doi_path='./Data/Datasets/Pranav/pranav_doi.txt', output_path='./Data/Datasets/Pranav/PDFs')
    # pdf_path = './Data/Datasets/Pranav/PDFs'
    # output_path = './Outputs/Pranav'

    pdf_path = './Data/biopolymer'
    output_path = './Outputs/biopolymer'
    
    pdfs = os.listdir(pdf_path)
    pdfs.sort()

    parsed_pdfs = os.listdir(output_path)

    for pdf in pdfs:
        print(pdf)
        # if pdf[:-4] not in parsed_pdfs:
        Parser(pdf_path=os.path.join(pdf_path, pdf), output_path=os.path.join(output_path, pdf[:-4]),extract_images=False, extract_molecular_image=False)
        tokenized_output_path = os.path.join(output_path, pdf[:-4], 'tokenized_paragraphs.txt')
        tokenized_sections = tokenize_passage(os.path.join(output_path, pdf[:-4], 'sections.txt'), output_path=tokenized_output_path)
        cdr_mentions = extractMentions(tokenized_output_path, 'ChemDataExtractor', PSC_MENTION_DICT, True)
        rule_mentions = extractMentions(tokenized_output_path, 'Rules', PSC_MENTION_DICT, True)
        all_mentions = []
        for i in range(len(cdr_mentions)):
            mentions_t = []
            mentions_t.extend(cdr_mentions[i])
            mentions_t.extend(rule_mentions[i])
            mentions_t = sorted(mentions_t, key=lambda x : x[0])
            all_mentions.append(mentions_t)
        with open(os.path.join(output_path, pdf[:-4], 'ner.txt'), 'w+') as file:
            #[[Sections[tuples]]]
            json.dump(all_mentions, file)
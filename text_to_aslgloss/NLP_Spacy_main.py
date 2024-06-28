"""
CODE DANS LA PERSPECTIVE D'UN DEPLOIEMENT STEAMLIT & Flask

Main code for NLP Spacy english-to-ASL gloss translation

Input: 1 sentence in english

Output: 1 sentence in ASL gloss

Model: NLP Spacy 

Pipeline with next process: ASL glosses-to-sequence of videos of keypoints:
voir dans le cadre d'une API Flask:
- 1. prompt : voir dans
- 2. english sentence -> sentence of ASL glosses
- 3. save   
"""

## Imports
import spacy
from spacy.scorer import Scorer
import pandas as pd

from ASL_gloss_functions import process_sentence

import argparse


## reference language
nlp = spacy.load("en_core_web_sm")


## custom class for translation
"""
set of rules for ASL conversion:
1. Uppercase Letters: Write each ASL sign in uppercase letters
2. Non-Manual Signals (NMS): Indicate non-manual signals such as facial expressions or body movements above the glossed sign.
3. Fingerspelling: Represent fingerspelled words with dashes between each letter.
4. Lexicalized Fingerspelling: Indicate lexicalized fingerspelling with a # symbol.
5. Repetition: Show repeated signs with a plus sign (+) after the gloss.
6. Role Shift: Indicate role shift with "rs" before the gloss.
7. Indexing/Pointing: Use "ix" followed by a subscript letter or number for indexing.
8. Directional Signs: Indicate the direction of the sign with arrows or other indicators.
9. Classifiers: Use abbreviations for classifiers.
10. Time Indicators: Place time indicators at the beginning of the sentence.
11. Topic-Comment Structure: Indicate the topic followed by the comment.
12. English Words/Concepts: Use English gloss in quotation marks for concepts without direct ASL equivalents.
"""
class NlpSpacyTranslator():
    def __init__(self, sentence):
        self.sentence = sentence

    def translate_to_gloss(self):
        """
        - doc: after nlp processing: I write a sentence for testing Today 17.05 p.m.
        - gloss: TODAY
        - generated_gloss: TODAY ix_1 I WRITE  SENTENCE FOR TEST 17.05 P.M.
        """
        doc = nlp(self.sentence)
        ##print(f'doc after nlp processing: {doc}')
        generated_gloss = process_sentence(doc) ## deterministic model = set of ASL-Gloss-rules functions
        print(f'generated_gloss: {generated_gloss}')
        return generated_gloss


def main(eng_sentence):
    eng_to_asl_translator = NlpSpacyTranslator(sentence=eng_sentence)
    generated_gloss = eng_to_asl_translator.translate_to_gloss()
    return generated_gloss


# This block will only execute if the script is run directly
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add an argument
    parser.add_argument('sentence', type=str, help='sentence in english')

    # Parse the arguments
    args = parser.parse_args()

    # Pass the argument to the main function
    main(args.sentence)

import spacy
import pickle
from nltk.corpus import wordnet


def load_spacy_values(filepath_model_spacy='model_spacy_synonyms', filepath_docs_spacy = 'dict_spacy_object.pkl'):
    '''
    Loads a spaCy model and a dictionary of spaCy Doc objects from a pickle file.

    Parameters
    ----------
    filepath_model_spacy : str
        The local path to the spaCy model used for synonym detection.

    filepath_docs_spacy : str
        The local path to the pickle file containing a dictionary where the keys are tokens 
        and the values are the corresponding spaCy Doc objects serialized as bytes.

    Returns
    -------
    nlp : spacy.language.Language
        The loaded spaCy language model.

    dict_docs_spacy : dict
        A dictionary where the keys are tokens (str) and the values are spaCy Doc objects, 
        reconstructed from the serialized bytes.
    '''
    
    # ---- Load the spaCy NLP model
    #
    nlp = spacy.load(filepath_model_spacy)
    
    # ---- Load pickle file and reconstruct the dictionary with tokens as keys and spaCy Doc objects as values
    #
    with open(filepath_docs_spacy, 'rb') as file:
        dict_docs_spacy_bytes = pickle.load(file)
    
    dict_docs_spacy = {key: spacy.tokens.Doc(nlp.vocab).from_bytes(doc_bytes) for key, doc_bytes in dict_docs_spacy_bytes.items()}
    
    return nlp, dict_docs_spacy


def find_antonyms(word):
    '''
    Generate a set of all the antonyms of a given word

    Parameters
    ----------
    word : str
        The word that we want to find the antonyms

    Returns
    -------
    antonyms : set of str
        A set of all the antonym detected using nltk and WordNet
    '''
    
    antonyms = set()

    # ---- Load all the set of synonyms of the word recorded from wordnet
    #
    syn_set = wordnet.synsets(word)

    # ---- Loop over each set of synonyms
    #
    for syn in syn_set:
        # ---- Loop over each synonym
        #
        for lemma in syn.lemmas():
            # ---- Add antonyms of the synonyms to the antonyms set
            #
            if lemma.antonyms():
                antonyms.add(lemma.antonyms()[0].name())

    return antonyms


def find_synonyms(word, model, dict_embedding, list_2000_tokens):
    '''
    Finds the most similar token to a given word.

    Parameters
    ----------
    word : str
        The word that we want to find the most similar word

    model : spacy.language.Language
        spaCy language model to use for the detection of the synonym
    
    dict_embedding: dict
        A dictionary where the keys are tokens (str) and the values are spaCy Doc objects

    list_2000_tokens : list of str
        A list of 2000 tokens against which the gloss will be checked.
    
    Returns
    -------
    most_similar_token : str
        The most similar token to the given word 
    '''

    # ---- Skip synonym detection if the word is already in the list_2000_token
    #
    if word in list_2000_tokens:
        return word
    else:
        # ---- Remove antonyms of the given word of the list_2000_tokens (a word and an antonym might be similar in embedding representation)
        #
        antonyms = find_antonyms(word)
        list_2000_tokens_less_antonyms = [token for token in list_2000_tokens if token not in antonyms]

        # ---- Generate a list of tuple (token, similarities values between the embedding of the given word and the embedding of each token of the list_2000_tokens)
        #
        word_embedding = model(word)
        similarities=[]
    
        for token in list_2000_tokens_less_antonyms:
            similarities.append((token, dict_embedding.get(token).similarity(word_embedding)))

        # ---- Extract the most similar token of the list
        #
        most_similar_token = sorted(similarities, key=lambda item: -item[1])[0][0]

        return most_similar_token
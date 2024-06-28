# Define a list of question adverbs
opened_question_adverbs = ["how", "when", "where", "why", "how much", "how many", "how often", "how long", "what", "which", "who", "whose", "whom"]

## time adverbs to be moved at the beginning of ASL Gloss sentences
time_words = ["yesterday", "today", "tomorrow"]

# ASL glossing rules implemented in functions
def gloss_word(word):
    return word.upper()

def handle_fingerspelling(word):
    return '-'.join(list(word.upper()))

def handle_lexicalized_fingerspelling(word):
    return f"#{word.upper()}"

def handle_repetition(word, count):
    return f"{word.upper()}{'+' * (count - 1)}" if count > 1 else word.upper()

def handle_role_shift(sentence):
    return f"rs {sentence}"

def handle_indexing(token, index):
    return f"ix_{index} {token.upper()}"

def gloss_sentence(doc):
    glossed_sentence = []
    for token in doc:
        glossed_word = gloss_word(token.text)
        glossed_sentence.append(glossed_word)
    return " ".join(glossed_sentence)

def add_time_indicator(gloss_sentence_):
    for word in gloss_sentence_:
        if word.text.lower() in time_words:
            return f"{word.text.upper()} {gloss_sentence_.replace(word.text.upper(), '').strip()}"
    return gloss_sentence_

## skip stop_words
def skip_stop_words(word):
    if word.lower() == 'the' or word.lower() == 'a':
        return ''
    else:
        return word

## doc est une liste de tokens
def question_type(doc):
    if doc[-1].text == '?':
        if doc[0].text.lower() in opened_question_adverbs:
            return "wh-question"
        else:
            return "yes-no-question"
    return None

# add question id as a prefix
def process_sentence(doc):
    nms = {
        "wh-question": "wh-q",
        "yes-no-question": "y/n-q"
    }
    
    classifiers = {
        "car": "CL:3",
        "person": "CL:1"
    }
    
    glossed_sentence = []
    for token in doc:
        ## utilize token.lemma_, not .text
        #word = token.text.lower()
        word = token.lemma_.lower()
        
        if word in ["i", "me"]:
            glossed_word = handle_indexing("I", 1)
        elif word in ["you"]:
            glossed_word = handle_indexing("YOU", 2)
        elif word in classifiers:
            glossed_word = classifiers[word]
        else:
            glossed_word = gloss_word(word)
        glossed_word = skip_stop_words(glossed_word)
        
        glossed_sentence.append(glossed_word)    

    for gloss in glossed_sentence:
        if gloss.lower() in time_words:
            # move gloss at beginning
            glossed_sentence.insert(0, glossed_sentence.pop(glossed_sentence.index(gloss)))
            break
        
    type_doc = question_type(doc)
    if type_doc != None:
        glossed_sentence.insert(0, nms[type_doc])
        
    return " ".join(glossed_sentence)

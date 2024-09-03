import display_gloss as dg
import synonyms_preprocess as sp
from NLP_Spacy_base_translator import NlpSpacyBaseTranslator 
from flask import Flask,  render_template, Response, request

# ---- Initialise Flask App
#
app = Flask(__name__)

# ---- Initialise data
#
nlp, dict_docs_spacy = sp.load_spacy_values()
dataset, list_2000_tokens = dg.load_data()

# ---- Render the homepage template
#
@app.route('/')
def index():

    return render_template('index.html')

# ---- Translate english input sentence into gloss sentence
#
@app.route('/translate/', methods=['POST'])
def result():
    
    if request.method == 'POST':
        # ---- Get the raw sentence and translate it to gloss
        #
        sentence = request.form['inputSentence']
        eng_to_asl_translator = NlpSpacyBaseTranslator(sentence=sentence)
        generated_gloss = eng_to_asl_translator.translate_to_gloss()
        gloss_list_lower = [gloss.lower() for gloss in generated_gloss.split() if gloss.isalnum() ]
        gloss_sentence_before_synonym = " ".join(gloss_list_lower)

        # ---- Substitute gloss tokens with synonyms if not in the common token list
        #
        gloss_list = [sp.find_synonyms(gloss, nlp, dict_docs_spacy, list_2000_tokens) for gloss in gloss_list_lower]
        gloss_sentence_after_synonym  = " ".join(gloss_list)

        # ---- Render the result template with both versions of the gloss sentence
        #
        return render_template('translate.html',\
                                sentence=sentence,\
                                gloss_sentence_before_synonym=gloss_sentence_before_synonym,\
                                gloss_sentence_after_synonym=gloss_sentence_after_synonym)

# ---- Generate video streaming from gloss_sentence
#
@app.route('/video_feed')
def video_feed():
    
    sentence = request.args.get('gloss_sentence_to_display', '')
    gloss_list = sentence.split()
    return Response(dg.generate_video(gloss_list, dataset, list_2000_tokens), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=5000, debug=True)

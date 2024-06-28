from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from rouge_score import rouge_scorer

## Question: les paires de phrases des deux sets doivent avoir la même longueur ? 

class NlpMetrics():
    def __init__(self, sentences_set1, sentences_set2):
        ##self.metric = metric
        self.sentences_set1 = sentences_set1
        self.sentences_set2 = sentences_set2
    
    def metric_blue(self):
        # Tokenize sentences
        tokenized_set1 = [s.split() for s in self.sentences_set1]
        tokenized_set2 = [s.split() for s in self.sentences_set2]

        print(f'tokenized_set1: {tokenized_set1}')

        # Compute BLEU scores
        bleu_scores = [sentence_bleu([ref], hyp) for ref, hyp in zip(tokenized_set1, tokenized_set2)]
        average_bleu = sum(bleu_scores) / len(bleu_scores)
        print(f"Average BLEU Score: {average_bleu}")
        return average_bleu
    
    def metric_red(self):
        # Initialize ROUGE
        rouge = Rouge()

        # Compute ROUGE scores
        scores = [rouge.get_scores(ref, hyp)[0] for ref, hyp in zip(self.sentences_set1, self.sentences_set2)]
        average_rouge = {
            "rouge-1": sum(score['rouge-1']['f'] for score in scores) / len(scores),
            "rouge-2": sum(score['rouge-2']['f'] for score in scores) / len(scores),
            "rouge-L": sum(score['rouge-l']['f'] for score in scores) / len(scores)
        }
        print(f"Average ROUGE Scores: {average_rouge}")
        return average_rouge
    
    def metric_rouge(self):
        # Initialize ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        # Compute ROUGE scores for each pair of sentences
        scores = [scorer.score(ref, hyp) for ref, hyp in zip(self.sentences_set1, self.sentences_set2)]
        print(f'metric_rouge scores: {scores}')

        # Average the ROUGE scores across all sentence pairs
        average_scores = {
            "rouge1": sum(score['rouge1'].fmeasure for score in scores) / len(scores),
            "rouge2": sum(score['rouge2'].fmeasure for score in scores) / len(scores),
            "rougeL": sum(score['rougeL'].fmeasure for score in scores) / len(scores)
        }

        print(f"Average ROUGE Scores: {average_scores}")
        return average_scores

    
    """
    def compute_metric(self):
        if self.metric.lower() == 'rouge':
            return self.metric_red()
        if self.metric.lower() == 'blue':
            return self.metric_blue()
    """
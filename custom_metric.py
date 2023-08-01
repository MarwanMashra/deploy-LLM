# from evaluate import EvaluationModule, EvaluationModuleInfo
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import datasets
import evaluate
import sacrebleu as scb

# https://huggingface.co/spaces/evaluate-metric/sacrebleu/blob/d94719691d29f7adf7151c8b1471de579a78a280/sacrebleu.py
class MyCustomMetric(evaluate.Metric):
    def _info(self) -> evaluate.MetricInfo:
        return evaluate.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=[datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Sequence(datasets.Value("string", id="sequence"), id="references"),
                    }
                )],
        )
    
    def _compute(self, predictions, references):
        if isinstance(references[0], str):
            references = [[ref] for ref in references]

        references_per_prediction = len(references[0])

        transformed_references = [[refs[i] for refs in references] for i in range(references_per_prediction)]
        output = scb.corpus_bleu(
            predictions,
            transformed_references,
        )
        return {
            "sacrebleu_score": output.score,
        }


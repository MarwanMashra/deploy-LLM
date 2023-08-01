# from evaluate import EvaluationModule, EvaluationModuleInfo
import datasets
import evaluate
import sacrebleu as scb

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


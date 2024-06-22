# code copied and modfied from `https://huggingface.co/spaces/evaluate-metric/glue/blob/main/glue.py`
""" Classification metric. """

import datasets
from sklearn.metrics import f1_score

import evaluate


def simple_accuracy(preds, labels):
    return float((preds == labels).mean())


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = float(f1_score(y_true=labels, y_pred=preds))
    return {
        "accuracy": acc,
        "f1": f1,
    }

class Glue(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("int64"),
                    "references": datasets.Value("int64"),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
            format="numpy",
        )

    def _compute(self, predictions, references):
        return acc_and_f1(predictions, references)
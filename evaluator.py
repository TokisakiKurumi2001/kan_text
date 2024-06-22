import torch
from TeKan import TeKANDataLoader, TeKANClassifierModel, TeKAN
from loguru import logger
from transformers import AutoModel
import evaluate
from tqdm import tqdm

def postprocess( predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    true_labels = labels
    true_predictions = predictions
    return true_predictions, true_labels

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    # dataloader
    logger.info('Loading test dataloader')
    dataloader_config = {
        "tok_pretrained_ck": "roberta-base",
        "valid_ratio": 0.1,
        "num_train_sample": 20_000,
        "max_length": 256,
    }
    tekan_dataloader = TeKANDataLoader(**dataloader_config)
    [test_dataloader] = tekan_dataloader.get_dataloader(batch_size=16, types=["test"])
    logger.success('Loading dataloader successfully')

    logger.info('Loading model ...')
    pretrained_model = AutoModel.from_pretrained('roberta-base')
    classifier_model = TeKAN.from_pretrained('tekan_ckpt')
    model = TeKANClassifierModel(pretrained_model, classifier_model)

    device = torch.device('cuda:0')
    model = model.to(device)
    model.eval()
    logger.success('Model load successfully')

    logger.info('Loading evaluation metrics ...')
    test_metric = evaluate.load('metrics/classification_metrics.py')

    logger.info('Testing ...')
    for batch in tqdm(test_dataloader, desc='Testing', unit='batch'):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        with torch.inference_mode():
            logits = model(**batch)
        predictions = logits.argmax(dim=-1)

        decoded_preds, decoded_labels = postprocess(predictions, labels)
        test_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    results = test_metric.compute()
    logger.debug(results)
    
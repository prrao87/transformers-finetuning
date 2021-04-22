import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from typing import List, Dict
from datasets import load_dataset
from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
)


class ModelFileNotFoundError(Exception):
    pass


class DistilbertPredictor:
    # Define class labels for easier interpretation
    classes = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

    def __init__(self, model_checkpoint: str) -> None:
        from transformers import AutoModelForSequenceClassification
        from transformers import AutoTokenizer

        self.model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def predict_proba(self, text: str) -> np.ndarray:
        model_inputs = self.tokenizer(
            text, max_length=512, padding=True, truncation=True, return_tensors="pt"
        )
        raw_outputs = self.model(**model_inputs).logits
        result = torch.softmax(raw_outputs, dim=1).tolist()[0]
        proba = np.array(result)
        return proba

    def predict(self, text: str) -> str:
        proba = self.predict_proba(text)
        pred = np.argmax(proba)
        label = self.classes[pred]
        return label


class DistilbertOnnxPredictor:
    # Define class labels for easier interpretation
    classes = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

    def __init__(self, path_to_model: str, intra_op_num_threads: int = 3) -> None:
        from transformers import AutoTokenizer

        # For now only use uncased model 
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.intra_op_num_threads = intra_op_num_threads
        try:
            self.model = self.create_model_for_provider(path_to_model)
        except ValueError:
            print(f"Could not find ONNX model file (.onnx) at {path_to_model}")
            raise ModelFileNotFoundError

    def create_model_for_provider(self, model_path: str) -> InferenceSession:
        "Create ONNX model based on provider (we use CPU by default)"
        # Few properties that might have an impact on performance
        options = SessionOptions()
        options.intra_op_num_threads = self.intra_op_num_threads
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load the model as a graph and prepare the backend
        session = InferenceSession(
            model_path, options, providers=["CPUExecutionProvider"]
        )
        session.disable_fallback()
        return session

    def get_onnx_inputs(self, text: str) -> Dict[List[int], List[int]]:
        "Input IDs after tokenization are provided as a numpy array"
        model_inputs = self.tokenizer(
            text, max_length=512, padding=True, truncation=True, return_tensors="pt"
        )
        inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}
        return inputs_onnx

    def predict_proba(self, text: str) -> np.ndarray:
        """
        Returns a numpy array of dimension (M x N) where M = no. of text samples
        and N = no. of class labels
        The value at each index represents the probability of each class label.
        """
        inputs_onnx = self.get_onnx_inputs(text)
        raw_outputs = self.model.run(None, inputs_onnx)
        proba = softmax(raw_outputs[0])[0]
        return proba

    def predict(self, text: str) -> str:
        """
        {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        """
        proba = self.predict_proba(text)
        pred = np.argmax(proba)
        label = self.classes[pred]
        return label


def f1_multiclass(y_true, y_pred):
    "Prediction accuracy (percentage) and F1 score"
    acc = accuracy_score(y_true, y_pred) * 100
    f1_macro = f1_score(y_true, y_pred, average="macro") * 100
    f1_micro = f1_score(y_true, y_pred, average="micro") * 100
    print(
        "\nAccuracy: {:.3f}\nMacro F1-score: {:.3f}\nMicro F1-score: {:.3f}".format(
            acc, f1_macro, f1_micro
        )
    )


def make_onnx_predictions(model):
    """
    Run ONNX predictor on AG News test set and check accuracy/F1-score
    """
    # Load AG News test set
    data = load_dataset(
        'ag_news',
        split={
            'test': 'test[:100%]',
        },
    )
    texts = data['test']['text']
    labels = data['test']['label']
    pred_labels = [model.predict(sample) for sample in tqdm(texts)]
    # Make true labels the same format as the predicted labels
    true_labels = [model.classes[item] for item in labels]

    f1_multiclass(true_labels, pred_labels)


if __name__ == "__main__":

    text = "Disgraced Greek Sprinters Drug Tested by WADA  ATHENS (Reuters) - Greek sprinters Costas Kenteris and  Katerina Thanou have been dope tested by doctors from the World  Anti-Doping Agency, an official said Tuesday."
    # Regular PyTorch inference example
    # path_to_model = "./model_agnews/model_pytorch"
    # model = DistilbertPredictor(path_to_model)

    # ONNX inference example
    path_to_model = "./model_agnews/model_onnx/model_onnx-optimized-quantized"
    model = DistilbertOnnxPredictor(path_to_model, intra_op_num_threads=3)

    # print(model.predict(text))
    make_onnx_predictions(model)


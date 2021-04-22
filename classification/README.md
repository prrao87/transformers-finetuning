# Classification
This section showcases benchmark classification results for various models from the 🤗 Transformers v4.x API.

## AG News dataset

The AG News dataset is downloaded and processed from the [Hugging Face Datasets repository](https://huggingface.co/docs/datasets/). AG is a collection of more than 1 million news articles. News articles have been gathered from more than 2000 news sources by ComeToMyHead in more than 1 year of activity. The training set contains 120,000 samples, while the test set contains 7,600 samples.

## Evaluation
The script `predictors.py` contains predictor classes for each model, and can be used to run evaluations. The results and the performance (timing numbers + accuracy/F1) are listed accordingly. **Note that the actual timing numbers may not be the same when run on a different machine**, as it depends on many factors.

### DistilBERT PyTorch (CPU)
```
$ time python3.9 predictors.py
100%|███████████████████████████████████████████████████████████████████| 7600/7600 [06:14<00:00, 20.28it/s]

Accuracy: 94.013
Macro F1-score: 94.014
Micro F1-score: 94.013
python3.9 predictors.py  1487.03s user 5.25s system 392% cpu 6:20.12 total
```

### DistilBERT ONNX (`intra_op_num_threads=2`)
```
$ time python3.9 predictors.py
100%|███████████████████████████████████████████████████████████████████| 7600/7600 [05:23<00:00, 23.51it/s]

Accuracy: 93.895
Macro F1-score: 93.888
Micro F1-score: 93.895
python3.9 predictors.py  646.98s user 1.26s system 196% cpu 5:29.91 total
```

### DistilBERT ONNX (`intra_op_num_threads=3`)
The `intra_op_num_threads` argument is quite influential in performance, so this number is upped to see if it helps with performance.

```
$ time python3.9 predictors.py
100%|███████████████████████████████████████████████████████████████████| 7600/7600 [04:04<00:00, 31.08it/s]

Accuracy: 93.895
Macro F1-score: 93.888
Micro F1-score: 93.895
python3.9 predictors.py  733.93s user 1.61s system 288% cpu 4:14.63 total
```

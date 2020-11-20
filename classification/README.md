# Classification

## AG News dataset

### DistilBERT vs. ALBERT

* DistilBERT uses a teacher-student model distillation approach to reduce the number of parameters in BERT. The resulting model has ~60M parameters, which is around half that of BERT-base.
* ALBERT uses a similar architecture as BERT, but optimizes the projection block through a factorized embedding matrix representation. In addition, ALBERT introduces parameter sharing across layers, increasing the compactness. These two design changes results in a 90% reduction in the number of parameters compared to BERT-base (ALBERT-base has just 12M parameters).

### Observations

* Training time is almost the same for BERT, DistilBERT and ALBERT (mostly because of similar batch sizes and other hyperparameters).
* The model size is significantly smaller than BERT, with ALBERT being the smallest in size (due to the much smaller number of parameters).
* Although ALBERT achieves significant levels of compression, its memory consumption is much higher than BERT (this could be because of the factorized embedding matrix and the way it is represented internally). It also looks like the PyTorch implementation of ALBERT is a little more memory-hungry than the TF version ([see this GitHub issue](https://github.com/huggingface/transformers/issues/2284)) - this is an ongoing issue and as a result, **ALBERT requires a smaller batch size** (8) as opposed to BERT and DistilBERT (16), with all other hyperparameters being the same.
* Overall, DistilBERT offers a good compromise of model size, training/inference time and accuracy when compared to BERT. However, [ALBERT v2](https://syncedreview.com/2020/01/03/google-releases-albert-v2-chinese-language-models/) was released in early 2020, which shows promise as being significantly better than v1, with accuracy levels comparable to DistilBERT on the GLUE benchmark, so this needs to be tested further on a range of cases and compared with BERT/DistilBERT.
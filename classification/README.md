# Classification

## AG News dataset

### DistilBERT vs. ALBERT

* DistilBERT uses a teacher-student model distillation approach to reduce the number of parameters in BERT. The resulting model has ~60M parameters, which is around half that of BERT-base.
* ALBERT uses a similar architecture as BERT, but optimizes the projection block through a factorized embedding matrix representation. In addition, ALBERT introduces parameter sharing across layers, increasing the compactness. These two design changes results in a 90% reduction in the number of unique parameters compared to BERT-base.
    * However, additional experiments showed that ALBERT has, in effect, more effective parameters than the equivalent BERT model. Because of parameter sharing, ALBERT stores roughly 1/10th the number of parameters as BERT, but during training/inference, the input tokens must pass through 12 encoder stacks (just like they do in BERT).
    * Since ALBERT-base has ~12 million unique parameters, and it has 12 encoders, the effective number of parameters (that participate in training/inference) is ~144 million parameters, which is 1.3x times that of BERT-base (which has 110M parameters).

### Observations

* ALBERT takes longer to train than BERT, and significantly longer to train than DistilBERT. This is despite the fact that, on paper, ALBERT is a "compressed" model. In reality, the number of effective parameters present in ALBERT (due to the 12 encoder stacks) is 12 times the number of parameters stated in the paper.
* Both DistilBERT and ALBERT are significantly smaller in size than BERT, with ALBERT being the smallest in size (due to the much smaller number of unique parameters being stored).
* Although ALBERT, on paper, achieves significant levels of compression, its memory consumption is much higher than BERT (this could be due to the factorized embedding matrix and the way it is represented internally). In addition, the PyTorch implementation of ALBERT seems to be a little more memory-hungry than the TF version ([see this GitHub issue](https://github.com/huggingface/transformers/issues/2284)) - this is an ongoing issue and as a result, **ALBERT requires a smaller batch size** (8) as opposed to BERT and DistilBERT (16), with all other hyperparameters being the same.
* Overall, DistilBERT offers the best compromise of model size, training/inference time and accuracy.
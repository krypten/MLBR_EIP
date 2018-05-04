<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

# Assignment 3

## Student Details
**Student Name:** Chaitanya Agrawal  
**Batch No.:** Batch 1

## Topics
### Cross-Entropy Loss Functions

Cross entropy loss indicates the distance between what the model believes the output distribution should be, and what the original distribution really is. Cross entropy formula given two distributions over discrete variable x, where q(x) is the estimate for true distribution p(x) is given by
 
\\( H(p,q) = -\sum_{\forall x} p(x) log(q(x)) \\)

So predicting a probability of .012 when the actual observation label is 1 would be bad and result in a high loss value. A perfect model would have a log loss of 0.

![cross_entropy](http://ml-cheatsheet.readthedocs.io/en/latest/_images/cross_entropy.png)

The graph above shows the range of possible loss values given a true observation. Cross-entropy loss increases as the predicted probability diverges from the actual label leading the neuron faster learning.

```python
# SKLearn
sklearn.metrics.log_loss(
    y_true,
    y_pred,
    eps=1e-15, normalize=True, sample_weight=None, labels=None)

# Tensorflow
tf.losses.softmax_cross_entropy(
    onehot_labels,
    logits,
    weights=1.0, label_smoothing=0, scope=None, loss_collection=tf.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
)
```

### Triplet-Loss Functions

Triplet embeddings consist of mapping a group of images to an embedding space, such that faces from the same person should be close together and form well separated clusters. A triplet is a 3-tuple of an anchor embedding, positive embedding (of the same person), and negative embedding (of a different person). Triplet loss is a way to learn good embeddings for each face. For some distance on the embedding space , the loss of a triplet that needs to be minimized is :

\\( L = \sum_{i}^{N}[||f(x_i^a) - f(x_i^p)||^{2} - ||f(x_i^a) - f(x_i^n)||^{2} + {\alpha}] \\)


Tensor flow API [triplet_semihard_loss](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/losses/metric_learning/triplet_semihard_loss)

**Simple implementation of triplet loss:**
```
anchor_output = ...    # shape [None, 128]
positive_output = ...  # shape [None, 128]
negative_output = ...  # shape [None, 128]

d_pos = tf.reduce_sum(tf.square(anchor_output - positive_output), 1)
d_neg = tf.reduce_sum(tf.square(anchor_output - negative_output), 1)

loss = tf.maximum(0.0, margin + d_pos - d_neg)
loss = tf.reduce_mean(loss)
```
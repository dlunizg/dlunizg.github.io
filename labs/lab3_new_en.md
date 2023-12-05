---
layout: page
mathjax: true
permalink: /lab3/
---

# Exercise 3: analysis of sentiment classification

In the third laboratory exercise,
we are addressing the problem of 
sentiment analysis of movie reviews
using recurrent neural networks.

We will use the
[Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/)
(SST) dataset for training.
This is a smaller dataset 
commonly used in natural language processing 
and it is specific because, in its full form,
it includes sentiment value annotations
for each node in the parsing tree of each instance.
Unfortunately, despite the comprehensive annotation of documents, 
this dataset is often criticized as overly simplistic, 
as the classification for most instances
boils down to recognizing keywords.

**Do not** download the dataset from the official repository
since it is in a tree structure format.
Preprocessed versions of the dataset are provided
[here](https://github.com/dlunizg/dlunizg.github.io/tree/master/data/lab3).
The data has been converted to lowercase
and some tokens have been filtered.


For the purposes of the laboratory exercise,
you also need to download a set of
[vector representations](https://drive.google.com/open?id=12mA5QEN4nFcxfEzOS8Nqj5afOmkuclc7)
for all words present in the train split of the SST dataset.

Your task in the third laboratory exercise is
to verify how straightforward the SST dataset actually is
and to evaluate recurrent neural networks
as well as alternative approaches based on summarization.
You are given a series of control outputs
throughout the exercise instructions,
and these should be the same in your implementation
unless stated otherwise.


### Stanford Sentiment Treebank dataset

The data in the text files is separated
by a comma and a single space: `, `.
A comma will not appear within a sample.
The data consists of the text of the input instance
and the target class label.
The class label is either `positive` or `negative`,
while the text of the input instance
is a sequence of pre-segmented tokens
separated by a single space.
The text of the input instance
and the class label is what we call **fields**.

An example of a sample in the test subset:

```
it 's a lovely film with lovely performances by buy and accorsi, positive
```

SST subsets may be found in three files:
`sst_train_raw.csv`,
`sst_valid_raw.csv`,
`sst_test_raw.csv`.
All three files use the same data format.

### Task 1: Data loading (25% of points)


Unlike computer vision and the `torchvision` package,
the natural language processing ecosystem is highly fragmented,
with various conflicting ideologies regarding data loading.
There are several libraries available for these purposes
(torchtext, AllenNLP, ...),
but we will not be using their functionalities
within the scope of this exercise.


To better understand the challenges of text analysis,
your first task is to implement the loading of the dataset from disk.
We will use classes built-in PyTorch
`torch.utils.data.DataLoader` and `torch.utils.data.Dataset`
for iterating and batching data.
When loading data,
we suggest you use a conceptual division
into the following three classes, whose naming **may differ**:

- The `Instance` class, which serves as a lightweight wrapper around the data.
A simple and useful way to implement such a class could be
[data classes](https://docs.python.org/3/library/dataclasses.html),
available from Python version 3.7 onward.
An alternative approach is to use
[named tuples](https://docs.python.org/3/library/collections.html#collections.namedtuple).

- The `NLPDataset` class, which should inherit `torch.utils.data.Dataset`
and implement the `__getitem__` method.
This class is designed for storing and retrieving data,
as well as operations that require the entire dataset,
such as building a vocabulary.

- The `Vocab` class, which converts textual data into indices, 
a process known as *numericalization*. We will analyze the functionality of the Vocab class in more detail in the next subsection.

#### The `Vocab` class
As mentioned in the lectures,
one of the hyperparameters
for any natural language processing model
is the choice of vocabulary size,
i.e., the number of words we will represent in our model.
In practice, the vocabulary selection process
is often carried out in some version of the `Vocab` class
during the construction of the `itos` (index-to-string)
and `stoi` (string-to-index) dictionaries.

For each input field in our dataset, 
we assign a vocabulary.
Your vocabulary implementation should be built
based on a dictionary of frequencies for a specific field.
The frequency dictionary contains
all the tokens that have appeared in that field,
with values representing the number of occurrences of each token.

An example of most common words and
their frequencies for the `train` subset,
which contains `14804` different tokens:
```
the, 5954
a, 4361
and, 3831
of, 3631
to, 2438
```

The convention is to assign lower indices
to more frequent words.
However, before we proceed with converting words to indices,
we need to deal with the concept of `special symbols`.

**Special symbols** are tokens that are **not** present
in our dataset but are essential for numericalizing our data.
Examples of these symbols that we will use 
in the laboratory exercise are
the padding token `<PAD>` and the unkonwn token `<UNK>`

The padding token is necessary to ensure that our batches 
(consisting of examples of different lengths) 
are of equal length,
while the unknown token is used for words
not in our vocabulary — either due to size limitations 
or because they did not appear frequently enough in the data.

Special symbols always have the lowest indices, 
and for consistency, we will assign index 0
to the padding token and index 1 to the unknown token.
The remaining words should be assigned indices based on their frequency,
following the principle that more frequent words have lower indices.
Special symbols are used **only in the text field**.


Examples of words and their indices our `stoi` dictionary
for the text field of our training vocabulary:
```
<PAD>: 0
<UNK>: 1
the: 2
a: 3
and: 4
my: 188
twists: 930
lets: 956
sports: 1275
amateurishly: 6818
```

The `stoi` dictionary for the target field vocabulary is quite short:
```
positive, 0
negative, 1
```

Your implementation of the `Vocab` class must implement the functionality of
converting a sequence of tokens (or a single token) into numbers.
You may implement this functionality in a method called `encode`.
An example of this conversion for the fourth sample in the training data.

```python
instance_text, instance_label = train_dataset.instances[3]
print(f"Text: {instance_text}")
print(f"Label: {instance_label}")
>>> Text: ['yet', 'the', 'act', 'is', 'still', 'charming', 'here']
>>> Label: positive
print(f"Numericalized text: {text_vocab.encode(instance_text)}")
print(f"Numericalized label: {label_vocab.encode(instance_label)}")
>>> Numericalized text: tensor([189,   2, 674,   7, 129, 348, 143])
>>> Numericalized label: tensor(0)
```

Furthermore, your implementation of the `Vocab` class should take the following parameters:

- `max_size`: the maximum number of tokens that is stored into the vocabulary (including the special symbols). `-1` denotes that all tokens should be considered.
- `min_freq`: the minimal frequency that a token should for it to be included into a vocabulary (\ge). Special symbols do not go through this check.

An example showing how to build a vocabulary that includes all of the tokens
(including the special symbols)

```python
text_vocab = Vocab(frequencies, max_size=-1, min_freq=0)
print(len(text_vocab.itos))
14806
```

**Important**: A vocabulary is built  **only** on the train set.
This vocabulary is 
then set for both validation and test subsets.
This is considered to be most principled approach in text analysis, as otherwise
we would have information leaking from validation and test data into
model training.
 
For example -- in a real-world situation, a deployed model is not likely
to have every possible word in its vocabulary. This approach is therefore
stricter, but also more realistic.

#### Loading vector representations

In addition to the training data, you are also given 
a set of pretrained word representations [GloVe](https://nlp.stanford.edu/projects/glove/). 
You may download these vector representations [here](https://drive.google.com/open?id=12mA5QEN4nFcxfEzOS8Nqj5afOmkuclc7).


Word vectors are stored in a text format,
 where each line contains a token (word) 
and its 300-dimensional vector representation. 
The word vectors used in the exercise will always be 300-dimensional. 
The elements of each row are separated by a comma.

An example of the first row:

```
the 0.04656 0.21318 -0.0074364 -0.45854 -0.035639 ...
```

Your task is to implement a function
that will generate an embedding matrix
for a given vocabulary (an iterable of strings).
Your function should support two methods 
of generating the embedding matrix:
random initialization from the standard normal distribution
\((\mathbb{N}(0,1))\) and loading from a file.

If you do not find a vector representation for a word
during file loading, initialize it using a normal distribution.
You **must** initialize the vector representation
for the padding token (at index 0) to a vector of zeros.


A simple way to implement this loading
is to initialize the matrix from the standard normal distribution
and then overwrite the initial representation for each word you successfully load.

**Important**: Ensure that the order of vector representations
in the matrix corresponds to the order of words in the vocabulary!
For example, at index 0, there must be a representation
for the special padding token.

Once you have successfully loaded your \\(V\times d\\) embedding matrix, 
use [`torch.nn.Embedding.from_pretrained()`](https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding.from_pretrained)
to store your matrix into an optimized vector representation wrapper
Set the parameter `padding_idx` to 0 (as this is the index for the padding symbol in your embedding matrix),
and set the `freeze` parameter to `True` if you are using pretrained representations, or to `False` otherwise.

#### Overriding `torch.utils.data.Dataset` methods

To complete the `NLPDataset` implementation,
you need to override the `__getitem__` method
which allows us to index our classes.
For the purposes of this exercise,
this method needs to return the numericalized text
and label of the referenced instance.
It is OK to have an "on-the-fly" numericalization,
and it is not necessary to cache anything.

An example of numericalization with overriding:

```python
instance_text, instance_label = train_dataset.instances[3]
# We reference a class attribute without calling the overriden method
print(f"Text: {instance_text}")
print(f"Label: {instance_label}")
>>> Text: ['yet', 'the', 'act', 'is', 'still', 'charming', 'here']
>>> Label: positive
numericalized_text, numericalized_label = train_dataset[3]
# We use the overriden indexing method
print(f"Numericalized text: {numericalized_text}")
print(f"Numericalized label: {numericalized_abel}")
>>> Numericalized text: tensor([189,   2, 674,   7, 129, 348, 143])
>>> Numericalized label: tensor(0)
```

#### Data batching implementation: the `collate` method


We are almost ready for the implementation of the model —
the only thing left is to implement the transformation of a sequence of samples
into a batch. Here, we encounter again the issue of variable dimensionality.

In its default implementation of the collate function,
the Pytorch `torch.utils.data.DataLoader` expects
that the elements of a batch are of equal length.
This is not the case with text,
so in practice, we need to implement our own collate function.


We first need to explain what the collate method does.
Broadly speaking, the collate fuction builds the batch tensor for a 
given list of samples.
You may find the detailed documentation [here](https://pytorch.org/docs/stable/data.html#dataloader-collate-fn).

Your implementation look something like this:

```python
def collate_fn(batch):
    """
    Arguments:
      Batch:
        list of Instances returned by `Dataset.__getitem__`.
    Returns:
      A tensor representing the input batch.
    """

    texts, labels = zip(*batch) # Assuming the instance is in tuple-like form
    lengths = torch.tensor([len(text) for text in texts]) # Needed for later
    # Process the text instances
    return texts, labels, lengths

```


**Important:** your collate funkction should also
return the lengths of the original, unpadded sentences.
We will use these lengths when implementing some of our models 


The task of our collate function will be
to pad the lengths of instances with the padding token
to match the length of the longest instance in the batch.
For this, refer to the
[torch.nn.utils.rnn.pad_sequence](https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pad_sequence) function.
Note that your implementation of the collate function
must know which index is used as the padding token.


Once we have implemented all of described classes, 
data loading might look something like this:

```python
def pad_collate_fn(batch, pad_index=0):
    #...
    pass
batch_size = 2 # Only for demonstrative purposes
shuffle = False # Only for demonstrative purposes
train_dataset = NLPDataset.from_file('data/sst_train_raw.csv')
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                              shuffle=shuffle, collate_fn=pad_collate_fn)
texts, labels, lengths = next(iter(train_data_loader))
print(f"Texts: {texts}")
print(f"Labels: {labels}")
print(f"Lengths: {lengths}")
>>> Texts: tensor([[   2,  554,    7, 2872,    6,   22,    2, 2873, 1236,    8,   96, 4800,
                       4,   10,   72,    8,  242,    6,   75,    3, 3576,   56, 3577,   34,
                    2022, 2874, 7123, 3578, 7124,   42,  779, 7125,    0,    0],
                   [   2, 2875, 2023, 4801,    5,    2, 3579,    5,    2, 2876, 4802,    7,
                      40,  829,   10,    3, 4803,    5,  627,   62,   27, 2877, 2024, 4804,
                     962,  715,    8, 7126,  555,    5, 7127, 4805,    8, 7128]])
>>> Labels: tensor([0, 0])
>>> Lengths: tensor([32, 34])
```

Set the batch size to a bigger number and the
shuffle flag to `True` in the real implementation.

### Task 2: Baseline model (25% points)


The first step in any machine learning task
should be the implementation of a baseline model.
The baseline model serves as an estimation
of the performance that our actual,
usually *more complex* model must surpass as a shallow stream.
Additionally, baseline models will indicate
the actual cost of executing more advanced models.


Your task in the laboratory exercise
is to implement a model that utilizes
*mean pooling* to eliminate the problematic variable dimension.
When applying mean pooling, immediately eliminate
the **entire** temporal dimension (the so-called *window* is of size T).

The baseline model you should implement should look like this:

```
avg_pool() -> fc(300, 150) -> ReLU() -> fc(150, 150) -> ReLU() -> fc(150,1)
```

We suggest you use the
[BCEWithLogitsLoss](https://pytorch.org/docs/stable/nn.html#bcewithlogitsloss),
since you do not have to apply the sigmoid with it.
Alternatively, you may set the output
dimensionality to the number of classes and
use the cross-entropy loss.
Both approaches may be used in 
practice, and the choice comes down
to personal preference.

Use [Adam](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam).
as the optimization algorithm.


**Implement** evaluation metrics to track model performance.
Besides the loss, we are interested in **accuracy**,
[**f1 score**](https://en.wikipedia.org/wiki/F1_score) and
 **confusion matrix**. 
Measure and print out validation performance metrics
after each epoch, and at the end of training
measure and print out the performance on the test set.

For comparison purposes, our implementation of
the baseline model, which uses all of the words
in the vocabulary (`max_size=-1, min_freq=1`),
pretrained representations, `seed=7052020`, `lr=1e-4`,
`batch_size=10` on the train set, achieves the following accuracy:

```

Epoch 1: valid accuracy = 64.031
Epoch 2: valid accuracy = 66.941
Epoch 3: valid accuracy = 72.268
Epoch 4: Valid accuracy = 75.563
Epoch 5: Valid accuracy = 78.199

Test accuracy = 77.646
```

You may set the random seed for Pytorch CPU operations
with `torch.manual_seed(seed)`. Similarly,
if you are using NumPy in your code,
make sure to set its seed with `np.random.seed(seed)`.
When running your code on the GPU,
take into account that the reproduciability is not guaranteed
due to the CUDNN optimization of recurrent nerual networks,
though there are workarounds if you sacrifice 
efficiency. You may learn more about this
[here](https://pytorch.org/docs/stable/notes/randomness.html#cudnn).

**Important:** As long as the results of your code
do not vary significantly across different runs,
the exact output numbers do not need to be perfectly identical.
To check the variance (i.e., stability) of your model,
run your final model at least **5** times
with the same hyperparameters but different seeds.
Record the results of each run (all mentioned metrics)
in an Excel spreadsheet, Word document,
or a similar format. In the comments,
also include the hyperparameters used for running the model.


#### Code organization for Pytorch models

Due to "syntactic sugar" which usually accompanies
PyTorch training and evaluation,
the training programme is usually separated into three
semantic parts:

1. Initialization
  - Argument parsing
  - Data loading
  - Network initialization
2. Training loop
  - `train` method that executes one pass over the training subset
 - Syntactic sugar:
        - `model.train()` - enables dropout
        - `model.zero_grad()` for each batch - delete previous gradients for parameters as it is not done automatically
        - `loss.backward()`-propagate loss towards parameters
        - **[Optional]** `torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)`- clip gradijents based on their norm
        - `optimizer.step()`- adjust parameters depending on optimizer choice and gradient values 
3. Evaluation loop:
  - `evaluate` method that executes one pass over evaluation or test subsets
  - Syntactic sugar:
        - `with torch.no_grad():`- gradients are not calculated (memory and time efficiency)
        - `model.eval()`- disables dropout

These methods might look something like this:

```python
def train(model, data, optimizer, criterion, args):
  model.train()
  for batch_num, batch in enumerate(data):
    model.zero_grad()
    # ...
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()
    # ...


def evaluate(model, data, criterion, args):
  model.eval()
  with torch.no_grad():
    for batch_num, batch in enumerate(data):
      # ...
      logits = model(x)
      loss = criterion(logits, y)
      # ...

def main(args):
  seed = args.seed
  np.random.seed(seed)
  torch.manual_seed(seed)

  train_dataset, valid_dataset, test_dataset = load_dataset(...)
  model = initialize_model(args, ...)

  criterion = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

  for epoch in range(args.epochs):
    train(...)
    evaluate(...)
```

### Task 3: Implementing a recurrent neural network (25% points)

After a successful implementation of your baseline model,
it is time to try out a model based on recurrent neural networks.
It is your task to implement a basic recurrent neural network
using a recurrent cell of **your choice**:
["Vanilla" RNN](https://pytorch.org/docs/master/generated/torch.nn.RNN.html#torch.nn.RNN),
[GRU](https://pytorch.org/docs/master/generated/torch.nn.GRU.html#torch.nn.GRU),
[LSTM](https://pytorch.org/docs/master/generated/torch.nn.LSTM.html#torch.nn.LSTM)].

Once you choose a cell,
go over its documentation to learn more about it.
Here we point out several important details

- An RNN `forward` method, regardles of the cell type,
returns (1) a sequence of hidden states of the last layer and
(2) the hidden state (i.e., hidden states in the case of LSTMs)
for all layers at the last time step.
Typically, for the decoder, 
you want to use the hidden state
from the last layer at the last time step.
In LSTMs, this corresponds to the `h` component
of the dual `(h, c)` hidden state.
- To speed up execution, the RNN input should be
in the `time-first` format (as it is faster to 
*iterate* over the first dimension of the tensor). 
Make sure to transpose the inputs before 
sending them to the RNN cell.
- The tensors that serve as inputs to RNN cells
are often "[packed](https://pytorch.org/docs/master/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence)".
Packing is the process of encapsulating a tensor
with the actual lengths of each element in the batch.
If you use packing, the RNN network will not unroll
for time steps that contain padding in batch elements.
Besides efficiency, this approach can also improve accuracy,
but this part is **not** a mandatory part of your implementation.
- Implement [gradient clipping](https://pytorch.org/docs/master/generated/torch.nn.utils.clip_grad_norm_.html#torch.nn.utils.clip_grad_norm_)
before applying the optimization step.

A basic model that uses one of the possible
RNN cells should look something like this:

```
rnn(150) -> rnn(150) -> fc(150, 150) -> ReLU() -> fc(150,1)
```

Your basic RNN cell model should be unidirectional and have two layers.
For a multi-layer RNN, utilize the `num_layers` argument
when constructing the RNN network.

For comparison, our implementation of a recurrent network with a GRU cell,
a full vocabulary (`max_size=-1, min_freq=1`) 
pretrained embeddings,
`seed=7052020`, `lr=1e-4`, `batch_size=10` and `gradient_clip=0.25`
achieves the following accuracy on the training set:

```
Epoch 1: valid accuracy = 67.930
Epoch 2: valid accuracy = 77.155
Epoch 3: valid accuracy = 78.254
Epoch 4: valid accuracy = 79.517
Epoch 5: valid accuracy = 80.011

Test accuracy = 79.985
```

Regardless of which cell you choose,
run the training process at least **5** times
with the same hyperparameters but different seeds.
Monitor all implemented metrics and record them in a file.

### Task 4: Model comparison and hyperparameter search (25% points)

As we can see, our initial model implementations
are quite similar in terms of accuracy.
Since the results of running a model
for a set of hyperparameters 
can be due to pure luck or misfortune,
in this part of the lab exercise,
we will implement an exhaustive search
through model variants and their hyperparameters.

#### RNN cell comparison

Regardless of which RNN cell you chose in the third task, 
extend your code in a way that the type of RNN cell is an argument.
Run your code for the remaining types of RNN cells and record the results.
Is there an obvious winner? Is there an obvious loser?

Repeat this comparison with a change in
the hyperparameters of recurrent neural networks.
The following hyperparameters of recurrent neural networks are of interest:

- hidden_size
- num_layers
- dropout: applied **between** consecutive RNN layers (works only if there are two or more layers)
- bidirectional: the dimensionality of the output of a bidirectional RNN cell is **doubled**

Try **at least** 3 different values for each hyperparameter
(except for bidirectionality, which has only two values).
The way you combine these values is entirely up to you
(exhaustive grid search is too time-consuming).
Run each cell type for each combination of hyperparameters
and record the results (relevant metrics).
Don't be afraid to make aggressive changes in 
the values of hyperparameters (small changes won't give you much information).
Do you notice that any hyperparameter
significantly affects the performance of the cells? Which one?

Remember/write down the set of hyperparameters
that gives you the best results.
For this set, run the training at least 
5 times with different seeds and record the obtained metrics.

#### Hyperparameter optimization

Try running the recurrent neural networks 
for the best set of hyperparameters 
without using pretrained vector representations. 
Do the same for your baseline model.
Which model is more affected by the loss of pretrained representations?

Input vector representations are a crucial hyperparameter,
for which, within the scope of the lab exercise,
we only have two values: whether we use them or not. 
In text analysis, input vector representations
are a significant part of the algorithm's success.

In this part of the lab exercise,
you need to select **at least 5** 
of the following hyperparameters 
and check how the models perform for their variations. 
If a hyperparameter affects both the baseline model 
and the recurrent neural network, 
run experiments on both models.

For the recurrent neural network cell,
choose the one that achieves (in your opinion)
better results from the previous part of the exercise.

For hyperparameters marked with a certain number of stars (\*),
choose **only one** of those with the same number of stars.

Potential hyperparameters:

- (\*) Vocabulary size **V**
- (\*) Minimal word frequency **min_freq**
- (\*\*) Learning rate
- (\*\*) Batch size
- Dropout
- Number of layers
- Hidden layer dimensionality
- Optimizer (try out something other than Adam)
- Activation function (in fully connected layers)
- Gradient clipping value
- Pooling type (in the baseline model)
- Freezing vector representation (`freeze` argument in the `from_pretrained` method)

For each of the selected hyperparameters,
try at least three different values (unless it's binary).
Record the results of the experiments. 
Run the baseline and recurrent models
with the best hyperparameters at least 
5 times and record the average and 
deviation of all tracked metrics. 
Does any parameter seem the most influential for success?
Don't be afraid to make aggressive changes
in the values of hyperparameters
as they will lead you to conclusions more easily.

#### Running experiments and tracking results

The method and format of recording results
are intentionally left open.
The goal of the exercise is not to force you
into a specific framework;
instead, consider this as a research project
where you need to present your findings
to someone and support your theses with numbers.
These numbers could be recorded in a table,
as part of free-form text with experiment descriptions,
or visualized. You will present the results verbally,
so choose the format that suits you best for that purpose.

**The number of epochs** is intentionally undefined
due to different hardware capacities on your personal computers.
The SST dataset is chosen precisely because it is small,
and even running on CPU should not take too much time.
Ideally, each experiment should be run for at least 5 epochs.
If your hardware does not allow this in terms of
time or space, please let us know.

### Bonus: attention (max 20% points)


The additional task will be the implementation 
of Bahdanau attention for sequence classification.
Specifically, we omit the query
from the attention formulation,
use hyperbolic tangent as activation,
and the hidden states (outputs of our recurrent neural network)
will simultaneously serve as keys and values.

Attention weights are calculated in the following manner:

$$
a^{(t)} = w_2 tanh(W_1 h^{(t)}) \quad \alpha = \text{softmax} (a)
$$

This are then used in weighted pooling
of hidden states in the following manner:

$$
out_{attn} = \sum_t^T \alpha_t h^{(t)}
$$

The results of the attention layer serve as the input of our output layer.

Use half of the dimensionality of the hidden state
of your recurrent neural network
as the dimension of the attention mechanism's
hidden state (output dimension of the matrix \[W_1\]).

Make the attention mechanism for 
the recurrent neural network optional in your 
implementation (make it a hyperparameter).
Check how attention contributes 
to the results of each neural cell. 
Run models with and without the attention mechanism 
and record the average and deviation 
of all monitored metrics.

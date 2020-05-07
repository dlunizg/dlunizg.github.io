---
layout: page
mathjax: true
permalink: /lab3en_old/
---

## Exercise 3: Sequence modelling with recurrent neural networks

In the third lab assignment our topics are recurrent neural networks and character-level language modellng.

As convolutional neural networks are somewhat designed for image processing through iteratively deepening local filters, recurrent neural networks are an architecture geared towards processing sequential data, where the sequential dimension is commonly referred to as "time". A simple motivating example is written language, where we can imagine the current meaning of a sentence after \\(k\\) words as a function of all previous words. The temporal dimension here would be along the reading order of the words in the sentence -- in the case of English -- left-to-right.

Language modelling is a problem of approximating the probability distribution over the vocabulary of words for a given context. For a sample sentence \\(\mathbf{w}\\) *"The quick brown fox jumped"*, when predicting the word *"jumped"*, the left-hand side (LHS) context of size 4 is *"The quick brown fox"*. Similarly, the context of size \\(k\\) would be \\(k\\) words to the left and the right of a center word. Based on the context, the language model has to determine he probability of each word in the vocabulary, and our aim is for *"jumped"* to be highly likely. The probabilites for each correct word in a sequence are then summarized by a measure called perplexity, which measures how likely the language model thinks the sequence is.

### Dataset: Cornell Movie--Dialogs Corpus

Your task in this laboratory exercise is to learn a language model in the domain of movie dialogue. The orginal dataset can be downloaded [here](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html), while the version used in the lab assignment is a subset of the complete dataset. Examples of movies from the subset can be seen below:

<p align="center">
<img src ="/assets/lab3/movie_covers.png" width="50%"/>
</p>

A sample dialogue from the dataset:

> **ICE:**  
> You're a hell of a flyer.  You can be my wingman any time.
>   
> **MAVERICK:**  
> No. You can be mine  

All of the conversation in the dataset follows the format as above. The name of the person in upper case, followed by a colon and a newline symbol, and then the text spoken by the person followed by two newline symbols. Dialogues of different movies and scenes aren't separated in any way.

Along with the subset of the dataset, a script `select_preprocess.py` is available, which selects a subset of titles found in a control text file, deals with the preprocessing and writes the dataset into a txt format. The script can be found [here](https://github.com/dlunizg/dlunizg.github.io/tree/master/code/lab3).

The characters left in the dataset after preprocessing are:

>  ! ' , . ` : ? 0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N O P Q R S T U V W X Y Z a b c d e f g h i j k l m n o p q r s t u v w x y z

along with a space and a newline symbol.

### Task 1: Data loading and batching

Your first task in the lab assignment is to load the text dataset and convert it into *minibatches*. You should implement the data loading class in a file named `dataset.py`. Any naming within the class is left up to you.

The data loading module should have the following arguments: the location of the dataset on disk (the `txt` file), the minibatch size and the sequence length (the number of unroll steps of the network). The module can conceptually be split into three parts:

#### 1.1: Data preprocessing

In order to use the textual data in recurrent neural networks, it is necessary to map them to a numeric space (in this case, a discrete one). We will do this by mapping each character to a unique identifier (*id*). In practice, the id-s are assigned by the "more frequent = smaller" principle, where the symbols which appear more often have a smaller id.

Your task is to implement a preprocessing function which computes the mapping (character - id) according to the aforementioned principle. Along with that, you should implement a function mapping a sequence of characters into a sequence of id-s, as well as a function mapping a sequence of id-s into a sequence of characters. The skeletons of the aforementioned functions might look as follows:

```python
# ...
# Code is nested in class definition, indentation is not representative.
# "np" stands for numpy.

def preprocess(self, input_file):
    with open(input_file, "r") as f:
        data = f.read().decode("utf-8") # python 2

    # count and sort most frequent characters

    # self.sorted chars contains just the characters ordered descending by frequency
    self.char2id = dict(zip(self.sorted_chars, range(len(self.sorted_chars)))) 
    # reverse the mapping
    self.id2char = {k:v for v,k in self.char2id.items()}
    # convert the data to ids
    self.x = np.array(list(map(self.char2id.get, data)))

def encode(self, sequence):
    # returns the sequence encoded as integers
    pass

def decode(self, encoded_sequence):
    # returns the sequence decoded as letters
    pass

# ...
```

#### 1.2: Splitting into minibatches

We aim to split our dataset into pairs of inputs and targets `x, y` of size `batch_size` x `sequence_length`. In the task of language modelling, our target is simply the input sequence, but shifted by one symbol -- our task is to predict the *next* character given a sequence of characters *so far*. Therefore, for the input \\(x_t\\) the target \\(y_t\\) we are trying to predict is actually \\(x_{t+1}\\). 

An example can be seen on the following image (from [*"The Unreasonable Effectiveness of Recurrent Neural Networks"*](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)):

<p align="center">
<img src ="/assets/lab3/charseq.jpeg" width="50%"/>
</p>

Therefore, your task in this step is to implement a function which, based on the existing preprocessed dataset, transforms that dataset into a sequence of batches of shape `batch_size` x `sequence_length`. The skeleton of this function may look as follows:

```python
# ...
# Code is nested in class definition, indentation is not representative.

def create_minibatches(self):
    self.num_batches = int(len(self.x) / (self.batch_size * self.sequence_length)) # calculate the number of batches

    # Is all the data going to be present in the batches? Why?
    # What happens if we select a batch size and sequence length larger than the length of the data?

    #######################################
    #       Convert data to batches       #
    #######################################

    pass
```

The concrete way of storing and representing batches is left up to you.

#### 1.3: Iterating over minibatches

The data loading class should also implement the logic for iterating over batches. Here, it is important to pay attention that the batches *shouldn't* be shuffled as we're working with textual data, and shuffling might ruin the natural ordering and information flow. Therefore, the order of batches in each epoch will be identical. A skeleton of this function may look as follows:


```python
# ...
# Code is nested in class definition, indentation is not representative.
def next_minibatch(self):
    # ...

    batch_x, batch_y = None, None
    # handling batch pointer & reset
    # new_epoch is a boolean indicating if the batch pointer was reset
    # in this function call
    return new_epoch, batch_x, batch_y

```

### Task 2: vanilla recurrent neural network

Recurrent neural networks are specialized towards processing directed sequences - but we haven't yet went over the way they work. This can easily be illustrated by the following graphs:


<p align="center">
<img src ="/assets/lab3/rnn_recurrence.png" width="20%"/>
<img src ="/assets/lab3/rnn_unrolled.png" width="50%"/>
</p>

In the first graph the recurrent neural network is written in a compact way with a recurrent relation (the black square symbolizes using values from the previous timestep), while the second graph demonstrates the unrolled recurrent network. It is important to note that the weight matrices (U, W, V) are **shared** across timesteps!

The mathematical expression being computed in the graphs is the following:

$$
 \mathbf{h}^{(t)} = tanh(\mathbf{Wh}^{(t-1)} + \mathbf{Ux}^{(t)} + \mathbf{b})
$$
\\
$$
 \mathbf{o^{(t)}} = \mathbf{Vh}^{(t)} + c
$$
\\
$$
 \mathbf{\hat{y}^{(t)}} = softmax(\mathbf{o^{(t)}})
$$

The initialization of the hidden state \\(h^(0)\\) is done by randomizing, zero-initialization or the initial hidden state is treated as a hyperparameter and trained alongside the model. In the scope of the lab assignment it is ok to merely initialize the hidden state to zeros.

The parameters of the network are, therefore, the matrices \\(U\\), \\(V\\) and \\(W\\) as well as the bias vectors \\(b\\) i \\(c\\).

In each timestep of language modelling we will perform a classiication problem (which is the next character?), and we will use the cross-entropy loss with which you have familiarized yourselves in the previous lab assignment.

As a reminder, the cross entopy loss and the computation of its gradient with respect to the softmax function is as follows:

$$
L = - \sum_{i=1}^{C} \mathbf{y}_i log(\mathbf{\hat{y}}_i) \\
$$

$$
\frac{∂L}{∂\mathbf{o}} = \mathbf{\hat{y}} - \mathbf{y} \\
$$

Where \\(\mathbf{\hat{y}}\\) is the vector of output probabilities of the network, and \\(\mathbf{y}\\) the true class distribution (a one-hot vector).

#### 2.1: Implementing the recurrent neural network

The vanilla neural network we will implement has three hyperparameters - the size of the hidden layer, the number of timesteps and the learning rate. The optimization algorithm which you will implement is Adagrad.

With the assumption of single precision of decimal numbers (4 bytes), and the size of the hidden layer of 500, the language modelling task with a vocabulary size of 70 (input and output dimension) - how large is the memory requirement of the parameters of the model? Which parameter takes up the most memory? Repeat the same exercise with a vocabulary of size 10000.

The aforementioned vocabulary sizes are the one you will use in the task of character level language modelling (approx. 70), and for word level language models (commonly much larger than \\(10^5\\)).

The initial hyperparameters are the hidden layer size of 100, an unroll of 30 timesteps and a learning rate of 1e-1. You should try out some other hyperparameters as well and report on loss and sample convergence.

The implementation of the recurrent neural network can be split conceptually into four parts, as follows:

##### 2.1.1: Parameter initialization

The parameter matrices should be initialized randomly from a gaussian distribution with a standard deviation of 1e-2. The bias vectors should be initialized to zeros. 

Parameter initializtion code could look as follows:

```python
# ...
# Code is nested in class definition, indentation is not representative.
# "np" stands for numpy
def __init__(self, hidden_size, sequence_length, vocab_size, learning_rate):
    self.hidden_size = hidden_size
    self.sequence_length = sequence_length
    self.vocab_size = vocab_size
    self.learning_rate = learning_rate

    self.U = None # ... input projection
    self.W = None # ... hidden-to-hidden projection
    self.b = None # ... input bias

    self.V = None # ... output projection
    self.c = None # ... output bias

    # memory of past gradients - rolling sum of squares for Adagrad
    self.memory_U, self.memory_W, self.memory_V = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
    self.memory_b, self.memory_c = np.zeros_like(self.b), np.zeros_like(self.c)

```

Remember to add a redundant dimension while initializing bias vectors for numpy broadcasting.

##### 2.1.2: Forward pass


The forward pass of the recurrent neural network can be imagined as a loop in which we iteratively perform a single timestep. The forward pass can therefore be implemented as a function which processes a single timestep, and a function which iterates over the whole sequence and stores the results. 

The skeletons of the aforementioned functions could look as follows:

```python
# ...
# Code is nested in class definition, indentation is not representative.

def rnn_step_forward(self, x, h_prev, U, W, b):
    # A single time step forward of a recurrent neural network with a 
    # hyperbolic tangent nonlinearity.

    # x - input data (minibatch size x input dimension)
    # h_prev - previous hidden state (minibatch size x hidden size)
    # U - input projection matrix (input dimension x hidden size)
    # W - hidden to hidden projection matrix (hidden size x hidden size)
    # b - bias of shape (hidden size x 1)

    h_current, cache = None, None
    
    # return the new hidden state and a tuple of values needed for the backward step

    return h_current, cache


def rnn_forward(self, x, h0, U, W, b):
    # Full unroll forward of the recurrent neural network with a 
    # hyperbolic tangent nonlinearity

    # x - input data for the whole time-series (minibatch size x sequence_length x input dimension)
    # h0 - initial hidden state (minibatch size x hidden size)
    # U - input projection matrix (input dimension x hidden size)
    # W - hidden to hidden projection matrix (hidden size x hidden size)
    # b - bias of shape (hidden size x 1)
    
    h, cache = None, None

    # return the hidden states for the whole time series (T+1) and a tuple of values needed for the backward step

    return h, cache

```

Notice that even though the parameters U, W and b exist as instance variables of the class (self.U, self.W, self.b), due to reproducibility we redundantly leave them as arguments of the functions. 

##### 2.1.3: Backward pass

The backward pass of the recurrent network is conceptually similar to the forward pass with the iteration being done in reverse, from the last timestep towards the first one. The backward pass is done by the backpropagation through time algorithm (BPTT). The skeletons of the BPTT algorithm functions could look as follows:


```python
# ...
# Code is nested in class definition, indentation is not representative.

def rnn_step_backward(self, grad_next, cache):
    # A single time step backward of a recurrent neural network with a 
    # hyperbolic tangent nonlinearity.

    # grad_next - upstream gradient of the loss with respect to the next hidden state and current output
    # cache - cached information from the forward pass

    dh_prev, dU, dW, db = None, None, None, None
    
    # compute and return gradients with respect to each parameter
    # HINT: you can use the chain rule to compute the derivative of the
    # hyperbolic tangent function and use it to compute the gradient
    # with respect to the remaining parameters

    return dh_prev, dU, dW, db


def rnn_backward(self, dh, cache):
    # Full unroll forward of the recurrent neural network with a 
    # hyperbolic tangent nonlinearity
    
    dU, dW, db = None, None, None

    # compute and return gradients with respect to each parameter
    # for the whole time series.
    # Why are we not computing the gradient with respect to inputs (x)?

    return dU, dW, db

```

In order to mitigate the exploding gradient problem, before applying the accumulated gradient, apply elementwise gradient clipping. We recommend you use the `np.clip()` method, and leave the values in the interval of \\([-5, 5]\\).

##### 2.1.4: Training loop

The training loop connects the recurrent network with the input data, calculates the loss and does parameter optimization. Ideally, the network should be idependent of the data - so it is recommended to separate the iteration over batches and epochs from the network class.

A skeleton implementation of the loss computation function could look as follows:

```python
# ...
# Code is nested in class definition, indentation is not representative.

def output(h, V, c):
    # Calculate the output probabilities of the network

def output_loss_and_grads(self, h, V, c, y):
    # Calculate the loss of the network for each of the outputs
    
    # h - hidden states of the network for each timestep. 
    #     the dimensionality of h is (batch size x sequence length x hidden size (the initial state is irrelevant for the output)
    # V - the output projection matrix of dimension hidden size x vocabulary size
    # c - the output bias of dimension vocabulary size x 1
    # y - the true class distribution - a tensor of dimension 
    #     batch_size x sequence_length x vocabulary size - you need to do this conversion prior to
    #     passing the argument. A fast way to create a one-hot vector from
    #     an id could be something like the following code:

    #   y[batch_id][timestep] = np.zeros((vocabulary_size, 1))
    #   y[batch_id][timestep][batch_y[timestep]] = 1

    #     where y might be a list or a dictionary.

    loss, dh, dV, dc = None, None, None, None
    # calculate the output (o) - unnormalized log probabilities of classes
    # calculate yhat - softmax of the output
    # calculate the cross-entropy loss
    # calculate the derivative of the cross-entropy softmax loss with respect to the output (o)
    # calculate the gradients with respect to the output parameters V and c
    # calculate the gradients with respect to the hidden layer h

    return loss, dh, dV, dc

```

The implementation of the parameter learning function could look as follows:

```python
# ...
# Code is nested in class definition, indentation is not representative.

# The inputs to the function are just indicative since the variables are mostly present as class properties

def update(self, dU, dW, db, dV, dc,
                 U, W, b, V, c,
                 memory_U, memory_W, memory_b, memory_V, memory_c):

    # update memory matrices
    # perform the Adagrad update of parameters
    pass

```

The implementation of the control flow loop and iterating the optimization process could look as follows:

```python
# ...
# code not necessarily nested in class definition

def run_language_model(dataset, max_epochs, hidden_size=100, sequence_length=30, learning_rate=1e-1, sample_every=100):
    
    vocab_size = len(dataset.sorted_chars)
    RNN = None # initialize the recurrent network

    current_epoch = 0 
    batch = 0

    h0 = np.zeros((hidden_size, 1))

    average_loss = 0

    while current_epoch < max_epochs: 
        e, x, y = dataset.next_minibatch()
        
        if e: 
            current_epoch += 1
            h0 = np.zeros((hidden_size, 1))
            # why do we reset the hidden state here?

        # One-hot transform the x and y batches
        x_oh, y_oh = None, None

        # Run the recurrent network on the current batch
        # Since we are using windows of a short length of characters,
        # the step function should return the hidden state at the end
        # of the unroll. You should then use that hidden state as the
        # input for the next minibatch. In this way, we artificially
        # preserve context between batches.
        loss, h0 = RNN.step(h0, x_oh, y_oh)

        if batch % sample_every == 0: 
            # run sampling (2.2)
            pass
        batch += 1

```

The `step` function, whose skeleton is not described, contains the logic of running a single forward and backward pass of a neural network on a minibatch of inputs. As inputs, the method accepts an initial hidden state vector, the one-hot encoded inputs and outputs of shapes BxTxV, where B is the minibatch size, T the number of timesteps to unroll the network for and V the size of the vocabulary. As outputs we receive the loss in that step as well as the hidden state from the **last timestep**, which should then be used as the next initial hidden state.

**Important**: Think about how the data has to be aligned in order for the last hidden states from the previous step to be valid initial hidden states for the current step.
 
#### 2.2: Sampling data from the learned network

When training the network, we implicitly take the next character as the next input instead of the one predicted by the network. This causes our loss to be artificially smaller, and is not applicable during testing (when we do not know the correct character).

In the scope of the lab assignment we will not dive deep into sequence sampling, but implement the bare minimum. Every `sample_every` minibatches, you should run the sampling method from our recurrent neural network in the following fashion:

1. Store the current hidden state of the network (`h_train`)
2. Initialize an empty hidden state for sampling (`h0_sample`)
3. Define a *seed* sequence of symbols to "warm up" the network - ex: `seed ='HAN:\nIs that good or bad?\n\n'`
4. Define how many characters you will sample (`n_sample=300`)
5. Run the forward pass on the seed sequence "seed"
6. Sample the remaining `n_sample` - `len(seed)` characters by using roulette-wheel (probability-proportional) sampling

The skeleton of the sampling function could look as follows:


```python
# ...
# code not necessarily nested in class definition
def sample(seed, n_sample):
    h0, seed_onehot, sample = None, None, None 
    # inicijalizirati h0 na vektor nula
    # seed string pretvoriti u one-hot reprezentaciju ulaza
    
    return sample

```

### Exercise 3: multilayer recurrent neural network in Tensorflow

TODO

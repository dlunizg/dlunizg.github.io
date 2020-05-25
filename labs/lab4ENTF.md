---
layout: page
mathjax: true
permalink: /lab4en_tf/
---
- [Generative models](#gm)
- [Restrictrd Boltzmann Machine](#rbm)
  - [Task 1](#1zad)
  - [Task 2](#2zad)
  - [Task 3](#3zad)
- [Varijationa Autoencoder](#vae)
  - [Task 4](#4zad)



<a name='gm'></a>

## Lab assignment 4: Generative Models (GM)
_v1.2-2018_

In this exercise, you will get to know generative models. Their main difference with respect to discriminative models is that they are designed to generate samples that are characteristic of the distribution of the samples used for training. In order for them to work properly, it is essential that they can learn the important characteristics of the samples from the training set. One possible representation of important characteristics is the distribution of input vectors, which a model could use to generate more probable samples (more frequent in the training set), and fewer samples that are less likely.

The distribution of samples from the training set can be described by the distribution of the probability of multiple variables
$$p(\mathbf x)$$. The probability of the training samples $$\mathbf x^{(i)}$$ should be high while the likelihood of other samples should be lower. Discriminative models, on the other hand, are focused on the posterior probability of the class $$ d $$.

$$ 
p(d\vert \mathbf{x})=\frac{p(d)p(\mathbf{x}\vert d)}{p(\mathbf{x})}
$$

The above expression suggests that knowledge of $$ p(\mathbf x)$$ could be useful information for discriminative models, though they generally don't use it directly. But one can expect that an accurate knowledge of $$ p(\mathbf x)$$ could help to better estimate $$ p(d \vert \mathbf{x}) $$. This idea is additionally supported by a reasonable assumption that both the input samples and the corresponding class $$ d $ (output) are the consequence of the same essential features. Input samples contain a substantial amount of essential information, but often also contain noise that makes it more difficult to model direct transformation to the output. The transformation of essential features to the output is presumably simpler than the direct input to output transformation.

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/bitneZen.svg" width="70%">
</div>

These ideas point to the use of generative models for extraction of essential features. Their primary purpose - generating samples - is not that important anymore. After the training, the layer containing the essential features can be used as an input layer for an additional discriminative model (eg. MLP). Such model would "more easily" produce the desired output. The focus of this exercise is on the training of generative models.

<a name='rbm'></a>

### Restricted Boltzmann Machine (RBM)

Boltzmann Machine (BM) is a [stochastic](https://en.wikipedia.org/wiki/Stochastic_neural_network) [recursive](https://en.wikipedia.org/wiki/Recursive_neural_network) [generative](https://en.wikipedia.org/wiki/Generative_model) network traind to maximize $$p(\mathbf x^{(i)})$$, and based on the Boltzmann distribution which assigns lover probability to states $$\mathbf x$$ with higher energy $$E(\mathbf x)$$ according to the following expression

$$
p(\mathbf{x})\propto
e^{\frac{-{E(\mathbf{x})}}{\mathit{kT}}}
$$

The product of Boltzmann constant $$ k $$ and the thermodynamic temperature $$ T $$ is ignored, or set to 1.

The states of individual BM nodes $$ x_j $$ are binary and can take values 0 and 1. The energy function $$ E (\mathbf x) $$ in BM is determined by the nodes'states  $$ x_j $$, the weights $ $ w_ {ji} $$ between them and the corresponding shifts $$ b_j $$.

$$
E(\mathbf{x})=-\left(\frac{1}{2}\sum _{i=1}^{N}\sum
_{\substack{j=1 \\ j\neq i}}^{N}w_{\mathit{ji}}x_{j}x_{i}+\sum
_{j=1}^{N}b_{j}x_{j}\right)=-\left(\frac{1}{2}\mathbf{x^{T}Wx}+\mathbf{b^{T}x}\right)
$$

The matrix $$\mathbf{W}$$ is symmetric and has zeros on the main diagonal. We define the probability of each sample as

$$
p(\mathbf{x};\mathbf{W},\mathbf{b})=\frac{e^{-E(\mathbf{x})/T}}{\sum_{\mathbf{x}}e^{-E(\mathbf{x})/T}}=\frac{e^{\frac{1}{2}\mathbf{x^{T}Wx}+\mathbf{b^{T}x}}}{Z(\mathbf{W},\mathbf{b})}
$$

$$Z(\mathbf W)$$ is called a partition function, and its role is to normalize the probability to make

$$
\sum_{\mathbf{x}}p(\mathbf{x};\mathbf{W},\mathbf{b})=1
$$

According to the selected energy function and Boltzmann's distribution, the probability of a node's state being 1 is equal to

$$
p(x_{j}=1)=\frac{1}{1+e^{-\sum
_{i=1}^{N}w_{\mathit{ji}}x_{i}-b_{j}}}=\sigma \left(\sum
_{i=1}^{N}w_{\mathit{ji}}x_{i}+b_{j}\right)
$$

In order for the BM energy function to describe higher-order correlations or more complex interconnections of individual elements of the data vector, we introduce the hidden variables $$h$$. Real data is then called the visible layer and is denoted by $$\mathbf in$$, while the hidden variables make the hidden layer $$\mathbf h$$.

$$
\mathbf{x}=(\mathbf v,\mathbf h)
$$

With RBMs, no interconnections are allowed within the same layer. This restriction (hence the name Restricted Boltzmann Machine) allows for simple updating of the network states. Although its purpose is well known, the hidden layer $$\mathbf h$$ and its distribution $$ p(\mathbf h)$$ are not predetermined.

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/rbm.svg" width="20%">
</div>

The energy of the network then becomes

$$
E(\mathbf{v},\mathbf{h})=-\mathbf{v^{T}Wh}-\mathbf{b^{T}h}-\mathbf{a^{T}v}
$$

The matrix $$\mathbf W$$ contains the weights connecting the visible and the hidden layer and is no longer symmetric. The vectors $$ \mathbf a$$ and $$ \mathbf b$$ contain the visible and hidden layers offsets.
According to the new structure and the previous equation for the probability of each element we get two equations for state update of RBM.


$$p(v_{i}=1)=\sigma \left(\sum
_{j=1}^{N}w_{\mathit{ji}}h_{j}+a_{i}\right)$$ za vidljivi sloj

$$p(h_{j}=1)=\sigma \left(\sum
_{i=1}^{N}w_{\mathit{ji}}v_{i}+b_{j}\right)$$ za skriveni sloj

The sampling of the values of a particular variable is carried out according to the above two equations and using a random number generator.


```python
def sample_prob(probs):
    """sampling of x according to probabilities p(x=1) = probs"""
    return tf.to_float(tf.random_uniform(tf.shape(probs)) <= probs)
```

**Training of the RBM-a**

Note that we want to maximize the likelihood of all training samples (input data) that are represented as the visible layer in an RBM. Therefore we maximize the product of all $$p(\mathbf {v}^{(j)})$$ where

$$
p(\mathbf{v};\mathbf{W},\mathbf{a},\mathbf{b})=\sum
_{\mathbf{h}}p(\mathbf{v},\mathbf{h};\mathbf{W},\mathbf{a},\mathbf{b})=\sum
_{\mathbf{h}}{\frac{e^{\mathbf{v}^{T}\mathbf{W}\mathbf{h}+\mathbf{b^{T}h}+\mathbf{a^{T}v}}}{Z(\mathbf{W},\mathbf{a, b})}}
$$

We can also maximize the logarithm of the probability of all visible vectors.

$$
\ln \left[\prod
_{n=1}^{N}p(\mathbf{v}^{(n)};\mathbf{W},\mathbf{a},\mathbf{b})\right]
$$

In order to achieve this, we need to determine the partial derivatives with respect to the network parameters

$$\frac{\partial }{\partial w_{\mathit{ij}}}\ln \left[\prod
_{n=1}^{N}p(\mathbf{v}^{(n)};\mathbf{W},\mathbf{a},\mathbf{b})\right]=\sum
_{n=1}^{N}\left[v_{i}^{(n)}h_{j}^{(n)}-\sum
_{\mathbf{v,h}}v_{i}h_{j}p(\mathbf{v,h};\mathbf{W},\mathbf{a},\mathbf{b})\right]=N\left[\langle
v_{i}h_{j}\rangle
_{P(\mathbf{h}\vert \mathbf{v}^{(n)};\mathbf{W},\mathbf{b})}-\langle
v_{i}h_{j}\rangle
_{P(\mathbf{v},\mathbf{h};\mathbf{W},\mathbf{a},\mathbf{b})}\right]$$

$$\frac{\partial }{\partial b_{j}}\ln \left[\prod
_{n=1}^{N}p(\mathbf{v}^{(n)};\mathbf{W},\mathbf{a},\mathbf{b})\right]=\sum
_{n=1}^{N}\left[h_{j}^{(n)}-\sum
_{\mathbf{v,h}}h_{j}p(\mathbf{v,h};\mathbf{W},\mathbf{a},\mathbf{b})\right]=N\left[\langle
h_{j}\rangle
_{P(\mathbf{h}\vert \mathbf{v}^{(n)};\mathbf{W},\mathbf{b})}-\langle
h_{j}\rangle
_{P(\mathbf{v},\mathbf{h};\mathbf{W},\mathbf{a},\mathbf{b})}\right]$$

$$\frac{\partial }{\partial a_{j}}\ln \left[\prod
_{n=1}^{N}p(\mathbf{v}^{(n)};\mathbf{W},\mathbf{a},\mathbf{b})\right]=\sum
_{n=1}^{N}\left[v_{j}^{(n)}-\sum
_{\mathbf{v,h}}v_{j}p(\mathbf{v,h};\mathbf{W},\mathbf{a},\mathbf{b})\right]=N\left[\langle
v_{j}\rangle -\langle v_{j}\rangle
_{P(\mathbf{v},\mathbf{h};\mathbf{W},\mathbf{a},\mathbf{b})}\right]$$

The final expressions of all three equations contain two components in which $$ \langle \rangle $$ brackets denote averaged values ​​for $$N$$ input samples (usually the mini batch size).
The first part of the final expressions refers to the states of the network when the input samples are fixed in the visible layer. To determine the corresponding states of the hidden layer $$\mathbf{h}$$, it is sufficient to determine each state $$h_j$$ according to the expression for $$p(h_j = 1)$$.
The second part refers to the state of the network without a fixed visible layer, so these states can be interpreted as something that the network imagines, based on the current configuration of the parameters ($$\mathbf W$$, $$\mathbf a $$ and $$\mathbf b$$). To get to these states we need to iteratively alternate recalculating new layer states ([Gibss sampling] (https://en.wikipedia.org/wiki/Gibbs_sampling)) according to the expressions for $$p(h_j = 1)$$ and $$p(v_i = 1)$$. Due to the absence of interconnections within the layers, all hidden elements are sampled at once, and then all the elements of the visible layer. Theoretically, the number of iterations should be very large to get what the network "really thinks" i.e. to get to the stationary distribution. The obtained state is then independent of the initial state. The practical solution to this problem is the Contrastive Divergence (CD) algorithm where it is sufficient to do only $$k$$ iterations (where $$k$$ is a small number, often only 1), and for the initial states of the visible layer, we take the input samples. Although this is a deviation from the theory, in practice it has proven to function well. The visualization of the CD-1 algorithm is given in the figure.

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/CDen.svg" width="50%">
</div>

Modification of weights and displacements for the input sample can then be callcualted in the followong way:

$$\Delta w_{\mathit{ij}}= \eta \left[\langle v_{i}h_{j}\rangle ^{0}-\langle
v_{i}h_{j}\rangle ^{1}\right]$$, 
$$\Delta b_{j}=\eta \left[\langle h_{j}\rangle ^{0}-\langle h_{j}\rangle
^{1}\right]$$, 
$$\Delta a_{i}=\eta \left[\langle v_{i}\rangle ^{0}-\langle v_{i}\rangle
^{1}\right]$$, 

The learning factor $$\eta$$ is usually set to a value less than 1. The first part of the expression for $$\Delta w_{\maths{ij}}$$ is often referred to as the positive phase, and a second part, as a negative phase.

The MNIST database will be used in the tasks. Although pixels in MNIST images can obtain real values from the range [0, 1], each pixel can be viewed as the probability of the stochastic binary variable $$p(v_i = 1)$$.
Input variables can then be treated as stochastic binary variables with [Bernoulli distribution] (https://en.wikipedia.org/wiki/Bernoulli_distribution), with given probability $$p(v_i=1)$$.

<a name='1zad'></a>

### Task 1

Implement the RBM that uses CD-1 for training. For input data use MNIST numbers. The visible layer must then have 784 elements, and the hidden layer should have 100 elements. Since the values of the input samples (image) are real numbers in the range [0 1], they can be used as $$p(v_i = 1)$$, so for the initial values of the visible layer, sampling should be performed. Set the mini batch size to 100 samples, and the number of epochs to 100.


**Subtasks:**
1. Visualize the weights of $$\mathbf W$$ obtained by training and try to interpret the weights associated with some hidden neurons.
2. Visualize the reconstruction results of the first 20 MNIST samples. Visualize the values of $$p(v_{i}=1)=\sigma \left(\sum_{j=1}^{N}w_{\mathit{ji}}h_{j}+a_{i}\right)$$ instead of the binary values ​​obtained by sampling.
3. Examine the activation frequency of hidden layer elements and visualize the learned weights of $$\mathbf W$$ sorted by the frequency
4. Skip the initial sampling/binarization based on the real input data, and use the original input data (real numbers from the range [0 1]) as input layer $$\mathbf v$$. How different is such RBM from the previous one?
5. Increase the number of Gibs sampling in CDs. What are the differences?
6. Examine the effects of varying the learning constant.
7. Randomly initialize the hidden layer, run a few Gibbs samplings, and visualize the generated visible layer
8. Perform above experiments with a smaller and a larger number of hidden neurons. What do you observe about weights and reconstructions?

Use the following template with the utility file [utils.py](/assets/lab4/utils.py).

**REMARK**: In addition to filling out the missing code, the template should be tailored as needed, and can be customized freely. So please **be especially careful with the claims that some of the code is not working for you!**

```python
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from utils import tile_raster_images
import math
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['image.cmap'] = 'jet'

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
    mnist.test.labels

def weights(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias(shape):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial)

def sample_prob(probs):
    """Sample vector x by probability vector p (x = 1) = probs"""
    return tf.to_float(tf.random_uniform(tf.shape(probs)) <= probs)

def draw_weights(W, shape, N, stat_shape, interpolation="bilinear"):
    """Visualization of weight
     W - weight vector
     shape - tuple dimensions for 2D weight display - usually input image dimensions, eg (28,28)
     N - number weight vectors
     shape_state - Dimension for 2D state display (eg for 100 states (10,10)
    """
    image = (tile_raster_images(
        X=W.T,
        img_shape=shape,
        tile_shape=(int(math.ceil(N/stat_shape[0])), stat_shape[0]),
        tile_spacing=(1, 1)))
    plt.figure(figsize=(10, 14))
    plt.imshow(image, interpolation=interpolation)
    plt.axis('off')
    
def draw_reconstructions(ins, outs, states, shape_in, shape_state, N):
    """Visualization of inputs and associated reconstructions and hidden layer states
     ins -- input vectors
     outs - reconstructed vectors
     states - hidden layer state vectors
     shape_in - dimension of input images eg (28,28)
     shape_state - Dimension for 2D state display (eg for 100 states (10,10)
     N - number of samples
    """
    plt.figure(figsize=(8, int(2 * N)))
    for i in range(N):
        plt.subplot(N, 4, 4*i + 1)
        plt.imshow(ins[i].reshape(shape_in), vmin=0, vmax=1, interpolation="nearest")
        plt.title("Test input")
        plt.axis('off')
        plt.subplot(N, 4, 4*i + 2)
        plt.imshow(outs[i][0:784].reshape(shape_in), vmin=0, vmax=1, interpolation="nearest")
        plt.title("Reconstruction")
        plt.axis('off')
        plt.subplot(N, 4, 4*i + 3)
        plt.imshow(states[i].reshape(shape_state), vmin=0, vmax=1, interpolation="nearest")
        plt.title("States")
        plt.axis('off')
    plt.tight_layout()

def draw_generated(stin, stout, gen, shape_gen, shape_state, N):
    """Visualization of initial hidden states, final hidden states and associated reconstructions
     stin - the initial hidden layer
     stout - reconstructed vectors
     gen - vector of hidden layer state
     shape_gen - dimensional input image eg (28,28)
     shape_state - Dimension for 2D state display (eg for 100 states (10,10)
     N - number of samples
    """
    plt.figure(figsize=(8, int(2 * N)))
    for i in range(N):

        plt.subplot(N, 4, 4*i + 1)
        plt.imshow(stin[i].reshape(shape_state), vmin=0, vmax=1, interpolation="nearest")
        plt.title("set state")
        plt.axis('off')
        plt.subplot(N, 4, 4*i + 2)
        plt.imshow(stout[i][0:784].reshape(shape_state), vmin=0, vmax=1, interpolation="nearest")
        plt.title("final state")
        plt.axis('off')
        plt.subplot(N, 4, 4*i + 3)
        plt.imshow(gen[i].reshape(shape_gen), vmin=0, vmax=1, interpolation="nearest")
        plt.title("generated visible")
        plt.axis('off')
    plt.tight_layout()
    
Nh = 100 # The number of elements of the first hidden layer
h1_shape = (10,10)
Nv = 784 # The number of elements of the first hidden layerBroj elemenata vidljivog sloja
v_shape = (28,28)
Nu = 5000 # Number of samples for visualization of reconstruction

gibbs_sampling_steps = 1
alpha = 0.1

g1 = tf.Graph()
with g1.as_default():
        
    X1 = tf.placeholder("float", [None, 784])
    w1 = weights([Nv, Nh])
    vb1 = bias([Nv])
    hb1 = bias([Nh])
    
    h0_prob = 
    h0 = sample_prob(h0_prob)
    h1 = h0

    for step in range(gibbs_sampling_steps):
        v1_prob = 
        v1 = 
        h1_prob = 
        h1 = 
        
    
    w1_positive_grad = 
    w1_negative_grad = 

    dw1 = (w1_positive_grad - w1_negative_grad) / tf.to_float(tf.shape(X1)[0])

    update_w1 = tf.assign_add(w1, alpha * dw1)
    update_vb1 = tf.assign_add(vb1, alpha * tf.reduce_mean(X1 - v1, 0))
    update_hb1 = tf.assign_add(hb1, alpha * tf.reduce_mean(h0 - h1, 0)) 

    out1 = (update_w1, update_vb1, update_hb1)
    
    v1_prob = 
    v1 = 
    
    err1 = X1 - v1_prob
    err_sum1 = tf.reduce_mean(err1 * err1)
    
    initialize1 = tf.global_variables_initializer()

batch_size = 100
epochs = 100
n_samples = mnist.train.num_examples
total_batch = int(n_samples / batch_size) * epochs

sess1 = tf.Session(graph=g1)
sess1.run(initialize1)

for i in range(total_batch):
    batch, label = mnist.train.next_batch(batch_size)
    err, _ = sess1.run([err_sum1, out1], feed_dict={X1: batch})
        
    if i%(int(total_batch/10)) == 0:
        print(i, err)

w1s = w1.eval(session=sess1)
vb1s = vb1.eval(session=sess1)
hb1s = hb1.eval(session=sess1)
vr, h1s = sess1.run([v1_prob, h1], feed_dict={X1: teX[0:Nu,:]})

# visualization of weights
draw_weights(w1s, v_shape, Nh, h1_shape)

# visualization of reconstructions and states
draw_reconstructions(teX, vr, h1s, v_shape, h1_shape, 200)

# visualization of a reconstructions with the gradual addition of the contributions of active hidden elements
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def draw_rec(inp, title, size, Nrows, in_a_row, j):
    """ Draw an iteration of creating the visible layer
     inp - visible layer
     title - thumbnail title
     size - 2D dimensions of visible layer
     Nrows - max. number of thumbnail rows 
     in-a-row. number of thumbnails in one row
     j - position of thumbnails in the grid
    """
    plt.subplot(Nrows, in_a_row, j)
    plt.imshow(inp.reshape(size), vmin=0, vmax=1, interpolation="nearest")
    plt.title(title)
    plt.axis('off')
    
    
def reconstruct(ind, states, orig, weights, biases):
    """ Sequential visualization of  the visible layer reconstruction
     ind - index of digits in orig (matrix with digits as lines)
     states - state vectors of input vectors
     orig - original input vectors
     weights - weight matrix
    """
    j = 1
    in_a_row = 6
    Nimg = states.shape[1] + 3
    Nrows = int(np.ceil(float(Nimg+2)/in_a_row))
    
    plt.figure(figsize=(12, 2*Nrows))
       
    draw_rec(states[ind], 'states', h1_shape, Nrows, in_a_row, j)
    j += 1
    draw_rec(orig[ind], 'input', v_shape, Nrows, in_a_row, j)
    
    reconstr = biases.copy()
    j += 1
    draw_rec(sigmoid(reconstr), 'biases', v_shape, Nrows, in_a_row, j)
    
    for i in range(Nh):
        if states[ind,i] > 0:
            j += 1
            reconstr = reconstr + weights[:,i]
            titl = '+= s' + str(i+1)
            draw_rec(sigmoid(reconstr), titl, v_shape, Nrows, in_a_row, j)
    plt.tight_layout()
    
reconstruct(0, h1s, teX, w1s, vb1s) # the first argument is the digit index in the digit matrix

# The probability that the hidden state is included through Nu input samples
plt.figure()
tmp = (h1s.sum(0)/h1s.shape[0]).reshape(h1_shape)
plt.imshow(tmp, vmin=0, vmax=1, interpolation="nearest")
plt.axis('off')
plt.colorbar()
plt.title('likelihood of the activation of certain neurons of the hidden layer')

# Visualization of weights sorted by frequency
tmp_ind = (-tmp).argsort(None)
draw_weights(w1s[:, tmp_ind], v_shape, Nh, h1_shape)
plt.title('Sorted weight matrices - from most to the least used')

# Generating samples from random vectors
r_input = np.random.rand(100, Nh)
r_input[r_input > 0.9] = 1 # percentage of active - vary freely
r_input[r_input < 1] = 0
r_input = r_input * 20 # Boosting in case a small percentage is active

s = 10
i = 0
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s

out_1 = sess1.run((v1), feed_dict={h0: r_input})

# Emulation of additional Gibbs sampling using feed_dict
for i in range(1000):
    out_1_prob, out_1, hout1 = sess1.run((v1_prob, v1, h1), feed_dict={X1: out_1})

draw_generated(r_input, hout1, out_1_prob, v_shape, h1_shape, 50)
```

<a name='2zad'></a>

### Task 2

Deep Belief Network (DBN) is a deep network that is obtained by stacking multiple RBMs one to another, each of which is trained greedily with inputs from the hidden ("outgoing") layer of the previous RBM (except the first RBM being trained directly with input samples). In theory, such DBN should increase $$ p(\mathbf v)$$ which is our initial goal. The use of DBN, ie reconstruction of the input sample, is carried out according to the scheme below. In the upward pass, hidden layers are determined from the visible layer until the highest RBM is reached, then the CD algorithm is executed, then in the downward direction, the lower hidden layers are determined until the visible layer. The weights between the individual layers are the same in the upward and in the downward pass. Implement a three-layer DBN that consists of two greedy RBMs. The first RBM should be as in 1st task, and the second RBM should have a hidden layer of 100 elements.

**Subtasks:**
1. Visualize the weights of $$\mathbf W_1$$ and $$\mathbf W_2$$ obtained by training.
2. Visualize the results of the reconstruction of the first 20 MNIST samples.
3. Randomly initialize the topmost hidden layer, run a few Gibbs samplings, and visualize generated visible layer patterns - compare with the previous task.
4. Set the number of hidden layer elements of the upper RBM to the number of lower RBM visible layer elements, and set the initial weights $$\mathbf W_2$$ to $$\mathbf W_1^T$$. What are the effects of change? Visualize elements of the topmost layer as 28x28 matrix.

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/DBN1.svg" width="100%">
</div>

Use the following template together with the template from the 1st task:

**REMARK**: In addition to filling out the missing code, the template should be tailored as needed, and can be customized freely. So please **be especially careful with the claims that some of the code is not working for you!**


```python
Nh2 = Nh # The number of elements of the second hidden layer
h2_shape = h1_shape 

gibbs_sampling_steps = 2
alpha = 0.1

g2 = tf.Graph()
with g2.as_default():
    
    X2 = tf.placeholder("float", [None, Nv])
    w1a = tf.Variable(w1s)
    vb1a = tf.Variable(vb1s)
    hb1a = tf.Variable(hb1s)
    w2 = weights([Nh, Nh2])
    hb2 = bias([Nh2])
    
    h1up_prob  = 
    h1up = 
    h2up_prob = 
    h2up = 
    h2down = h2up
    
    for step in range(gibbs_sampling_steps):
        h1down_prob = 
        h1down = 
        h2down_prob = 
        h2down = 
    
    w2_positive_grad = 
    w2_negative_grad = 

    dw2 = (w2_positive_grad - w2_negative_grad) / tf.to_float(tf.shape(h1up)[0])

    update_w2 = tf.assign_add(w2, alpha * dw2)
    update_hb1a = tf.assign_add(hb1a, alpha * tf.reduce_mean(h1up - h1down, 0))
    update_hb2 = tf.assign_add(hb2, alpha * tf.reduce_mean(h2up - h2down, 0))

    out2 = (update_w2, update_hb1a, update_hb2)

    # Reconstruction of the input based on the topmost hidden layer h3
    # ...
    # ...
    v_out_prob = 
    v_out = 
    
    err2 = X2 - v_out_prob
    err_sum2 = tf.reduce_mean(err2 * err2)
    
    initialize2 = tf.global_variables_initializer()

batch_size = 100
epochs = 100
n_samples = mnist.train.num_examples

total_batch = int(n_samples / batch_size) * epochs

sess2 = tf.Session(graph=g2)
sess2.run(initialize2)
for i in range(total_batch):
    # training iterations
    #...
    #...
    if i%(int(total_batch/10)) == 0:
        print(i, err)
        
    w2s, hb1as, hb2s = sess2.run([w2, hb1a, hb2], feed_dict={X2: batch})
    vr2, h2downs = sess2.run([v_out_prob, h2down], feed_dict={X2: teX[0:Nu,:]})

# visualization of weights
draw_weights(w2s, h1_shape, Nh2, h2_shape, interpolation="nearest")

# visualization of reconstruction and states
draw_reconstructions(teX, vr2, h2downs, v_shape, h2_shape, 200)

# Generating samples from random vectors of the topmost layer
#...
#...
# Emulation of additional Gibbs samplings using feed_dict
#...
#...
```

In order to further improve the generative properties of the DBN, the generative fine-tuning of the network parameters can be implemented. In the 2nd task, during reconstruction, the same weights and biases were used in the downward and upward steps. With fine-tuning, parameters that connect all layers except the two topmost, are split into two sets. The weight matrix between the lower layers is split into: $$\mathbf R_n$$ for the upward pass and $$\mathbf W'_n$$ for the downward pass. Initially, both matrices are equal to the original matrix $$\mathbf W_n$$. The new states of an upper hidden layer $$\mathbf s^{(n)}$$ are determined using $$\mathbf R$$ from the lower states $$\mathbf s^{(n-1)}$$ by sampling ($$sample \left(\sigma \left(\mathbf R_n \mathbf s^{(n-1)} + \mathbf b^{up}_n\right)\right) \to \mathbf s^{(n)}$$). In the downward pass (sleep phase) the "reconstruction" of the lower states $$\mathbf s^{(n-1)} $$ from $$\mathbf s^{(n)}$$ and the matrix $$\mathbf W'$$ ($$sample \left( \sigma \left(\mathbf W'_n \mathbf s^{(n)} + \mathbf b^{down}_{n-1} \right) \right) \to \mathbf s^{(n-1)}$$). The top two layers are classic RBM and share the same weight matrix for both directions, and the modification of these weights is carried out in the same way as in the 1st task.

Weight training between the lower layers is different. Matrices $$\mathbf W'_n$$ are modified when new states are determined using the matrix $$\mathbf R_n$$ in the upward pass. In the downward pass, the matrix $$\mathbf R_n$$ is modified. The bias vectors of the individual layers $$\mathbf b_n$$ are also split to the version for the upward pass $$\mathbf b_n^{up}$$ and for the downward pass $$\mathbf b_n^{down}$$. Initial biases are the same as the original ones $$\mathbf b$$.

To modify matrix $$\mathbf W'_n$$ in the upward pass ($$sample \left(\sigma \left(\mathbf R_n \mathbf s^{(n-1)} + \mathbf b^{up}_n\right)\right) \to \mathbf s^{(n)}$$) a downward sampling is performed $$sample \left( \sigma \left(\mathbf W'_n \mathbf s^{(n)} + \mathbf b^{down}_{n-1} \right) \right) \to \mathbf s^{(n-1)novo}$$. Weights are modified in the following way
$$\Delta w'_{\mathit{ij}}=\eta
s_{j}^{(n)}(s_{i}^{(n-1)}-s_{i}^{(n-1)\mathit{novo}})$$
Modification of downward biases is performed in the following way 
$$\Delta b_{\mathit{i}}^{\mathit{down}}=\eta
(s_{i}^{(n-1)}-s_{i}^{(n-1)\mathit{novo}})$$

To modify matrix $$\mathbf R_n$$ in the downward pass ($$sample \left( \sigma \left(\mathbf W'_n \mathbf s^{(n)} + \mathbf b^{down}_{n-1} \right) \right) \to \mathbf s^{(n-1)}$$) an upward sampling is performed $$sample \left(\sigma \left(\mathbf R_n \mathbf s^{(n-1)} + \mathbf b^{up}_n\right)\right) \to \mathbf s^{(n)novo}$$. Weights are modified in the following way
$$\Delta r_{\mathit{ij}}=\eta
s_{i}^{(n-1)}(s_{j}^{(n)}-s_{j}^{(n)\mathit{novo}})$$
Modification of upward biases is performed in the following way
$$\Delta b_{\mathit{i}}^{\mathit{up}}=\eta
(s_{i}^{(n)}-s_{i}^{(n)\mathit{novo}})$$

This procedure is performed for each training sample and is referred to as the up-down algorithm (sometimes wake-sleep algorithm).

HINT: pseudocode for training a four-layer DBN is available in the appendix of this [article](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)



<a name='3zad'></a>

### Task 3

Implement the generative fine-tuning procedure for DBN from the 2nd task. Use the CD-2 to train the topmost RBM.

Implementirajte postupak generativnog fine-tuninga na DBN iz 2. zadatka. Za treniranje gronjeg RBM-a koristite CD-2.

**Subtasks:**

1. Visualize the final versions of the matrix $$\mathbf W'$$ and $$\mathbf R$$.
2. Visualize the results of the reconstruction of the first 20 MNIST samples.
3. Randomly initialize the topmost hidden layer, run several Gibbs samplings, and visualize the generated visible layer - compare with previous tasks

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/DBN2.svg" width="40%">
</div>

Use the following template, as well as templates from tasks 1 and 2.

**REMARK**: In addition to filling out the missing code, the template should be tailored as needed, and can be customized freely. So please **be especially careful with the claims that some of the code is not working for you!**

```python
#
beta = 0.01

g3 = tf.Graph()
with g3.as_default():

    X3 = tf.placeholder("float", [None, Nv])
    r1_up = tf.Variable(w1s)
    w1_down = tf.Variable(tf.transpose(w1s))
    w2a = tf.Variable(w2s)
    hb1_up = tf.Variable(hb1s)
    hb1_down = tf.Variable(hb1as)
    vb1_down = tf.Variable(vb1s)
    hb2a = tf.Variable(hb2s)
    
    # wake pass
    h1_up_prob = 
    h1_up = # s^{(n)} in instructions
    v1_up_down_prob = 
    v1_up_down = # s^{(n-1)\mathit{novo}} in instructions
    
    
    # top RBM Gibs passes
    h2_up_prob = edid
    h2_up = 
    h2_down = h2_up
    for step in range(gibbs_sampling_steps):
        h1_down_prob = 
        h1_down = 
        h2_down_prob = 
        h2_down = 
       
    # sleep pass
    v1_down_prob = 
    v1_down = # s^{(n-1)} in instructions
    h1_down_up_prob = 
    h1_down_up = # s^{(n)\mathit{novo}} in instructions
    
    
    # generative weights update during wake pass
    update_w1_down = tf.assign_add(w1_down, beta * tf.matmul(tf.transpose(h1_up), X3 - v1_up_down_prob) / tf.to_float(tf.shape(X3)[0]))
    update_vb1_down = tf.assign_add(vb1_down, beta * tf.reduce_mean(X3 - v1_up_down_prob, 0))
    
    # top RBM update
    w2_positive_grad = 
    w2_negative_grad = 
    dw3 = 
    update_w2 = tf.assign_add(w2a, beta * dw3)
    update_hb1_down = tf.assign_add(hb1_down, beta * tf.reduce_mean(h1_up - h1_down, 0))
    update_hb2 = tf.assign_add(hb2a, beta * tf.reduce_mean(h2_up - h2_down, 0))
    
    # recognition weights update during sleep pass
    update_r1_up = tf.assign_add(r1_up, beta * tf.matmul(tf.transpose(v1_down_prob), h1_down - h1_down_up) / tf.to_float(tf.shape(X3)[0]))
    update_hb1_up = tf.assign_add(hb1_up, beta * tf.reduce_mean(h1_down - h1_down_up, 0))
    
    out3 = (update_w1_down, update_vb1_down, update_w2, update_hb1_down, update_hb2, update_r1_up, update_hb1_up)
    
    err3 = X3 - v1_down_prob
    err_sum3 = tf.reduce_mean(err3 * err3)
    
    initialize3 = tf.global_variables_initializer()

batch_size = 100
epochs = 100
n_samples = mnist.train.num_examples

total_batch = int(n_samples / batch_size) * epochs

sess3 = tf.Session(graph=g3)
sess3.run(initialize3)
for i in range(total_batch):
    #...
    err, _ = sess3.run([err_sum3, out3], feed_dict={X3: batch})
        
    if i%(int(total_batch/10)) == 0:
        print(i, err)
    
    w2ss, r1_ups, w1_downs, hb2ss, hb1_ups, hb1_downs, vb1_downs = sess3.run(
        [w2a, r1_up, w1_down, hb2a, hb1_up, hb1_down, vb1_down], feed_dict={X3: batch})
    vr3, h2_downs, h2_down_probs = sess3.run([v1_down_prob, h2_down, h2_down_prob], feed_dict={X3: teX[0:Nu,:]})


# visualization of weights
draw_weights(r1_ups, v_shape, Nh, h1_shape)
draw_weights(w1_downs.T, v_shape, Nh, h1_shape)
draw_weights(w2ss, h1_shape, Nh2, h2_shape, interpolation="nearest")

# visualization of reconstruction and states
Npics = 5
plt.figure(figsize=(8, 12*4))
for i in range(20):

    plt.subplot(20, Npics, Npics*i + 1)
    plt.imshow(teX[i].reshape(v_shape), vmin=0, vmax=1)
    plt.title("Test input")
    plt.subplot(20, Npics, Npics*i + 2)
    plt.imshow(vr[i][0:784].reshape(v_shape), vmin=0, vmax=1)
    plt.title("Reconstruction 1")
    plt.subplot(20, Npics, Npics*i + 3)
    plt.imshow(vr2[i][0:784].reshape(v_shape), vmin=0, vmax=1)
    plt.title("Reconstruction 2")
    plt.subplot(20, Npics, Npics*i + 4)
    plt.imshow(vr3[i][0:784].reshape(v_shape), vmin=0, vmax=1)
    plt.title("Reconstruction 3")
    plt.subplot(20, Npics, Npics*i + 5)
    plt.imshow(h2_downs[i][0:Nh2].reshape(h2_shape), vmin=0, vmax=1, interpolation="nearest")
    plt.title("Top states 3")
plt.tight_layout()

# Generating samples from random vectors of the topmost hidden layer
#...
#...
# Emulation of additional Gibbs sampling using feed_dict
#...
#...
```


<a name='vae'></a>

### Variational Autoencoder (VAE)

Autoencoder is a feed-forward network that uses backpropagation for training and can have a deep structure. Autoencoders are generative networks with a characteristic two-layer structure. The first part is called the encoder and maps (encodes) the input layer to the hidden layer. The second part is the decoder and transforms the hidden layer to the output layer. The primary goal of such a network is to achieve the similarity of inputs and outputs for each training sample, maximizing some similarity metrics. The primary goal is simple, which makes autoencoders' training unsupervised. The goal can be easily reached by direct copying the input to the output, but this is not in line with the hidden goal. The hidden goal, which is actually the most important, is to learn the essential features of the training samples. To achieve this, and avoid direct copying, various regularization techniques are used. Alternatively, another success rate can be used, such as maximizing probabilities. In any case, the variables of the hidden layer $$\mathbf z$ are responsible for extracting essential features from the input samples.

[Variational Autoencoders (VAE)](http://arxiv.org/abs/1312.6114) are autoencoders that maximize the probability of $$p(\mathbf x)$$ of all training samples $$\mathbf x^{(i)}$$. VAE does not use additional regularization techniques, but some of them can be included in VAE (eg a combination of VAE and denoising autoencoder gives better results). Important features in the hidden layer $$\mathbf z$$ then have the role in modeling $$p(\mathbf x)$$.

$$
p(\mathbf{x})=\int
p(\mathbf{x}\vert \mathbf{z};\theta
)p(\mathbf{z})\mathbf{\mathit{dz}}
$$

The probabilities on the right side of the equation are as unknown as is $$p(\mathbf x)$$, but we will approximate them with Gaussian distributions. $$\Theta$$ are model parameters and we determine them through the training process. An additional aim is to minimize the number of sampling operations, which are usually necessary when estimating unknown distributions.
In this case, it is best to maximize the log probability.

$$
\log _{\mathbf{\theta }}p(\mathbf{x}^{(1)},\ldots
,\mathbf{x}^{(N)})=\sum _{i=1}^{N}\log
_{\mathbf{\theta }}p(\mathbf{x}^{(i)})
$$

For $$p(\mathbf z)$$ we choose a normal distribution

$$
p(z)=N(0,1)
$$

By doing this, we seemingly limit the hidden layer in representing the important features.

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/VAE1.svg" width="50%">
</div>

The decoder models the conditional distribution $$p(\mathbf x \vert \mathbf z)$$ as a normal distribution and the parameters of that distribution are determined by the parameters of the network $$\Theta$$.


$$
p_{\mathbf{\theta }}(x\vert z)=N(\mu _{x}(z),\sigma _{x}(z))
$$

Parameters $$\Theta$$ include the weights and biases of all neurons of the hidden (internal) layers of the decoder and are determined through the training. The complexity of $$p(\mathbf x \vert \mathbf z)$$ depends on the number of hidden layers and the number of neurons in them. The dotted lines on the diagram indicate sampling operations. For the sake of visibility, only single variable z is shown in the diagram, instead of the vector of the hidden variables $$\mathbf z$$.

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/VAE_dec.svg" width="50%">
  <div class="figcaption figcenter">Decoder part</div>
</div>

Now we should determine $$p(\mathbf z \vert \mathbf x)$$, such that the above assumptions work well. Since we do not have a way to determine the appropriate $$p(\mathbf z \vert \mathbf x) $$, we will approximate it with the normal distribution $$q(\mathbf z \vert \mathbf x)$$, but we will carefully determine the parameters of this substitute distribution.

$$
q_{\mathbf{\phi
}}(\mathbf{z}\vert \mathbf{x})=N(\mathbf{\mu
_{z}(x),\sigma _{z}(x)})
$$

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/VAE_enc.svg" width="50%">
  <div class="figcaption figcenter">Encoder part</div>
</div>

Similar to the decoder, the parameters $$\Phi$$ include the weights and the biases of the encoder layers and they are determined by the training process. The complexity of $$q(\mathbf z \vert \mathbf x) $$ depends on the number of hidden layers and the number of neurons in them.

The model is now complete, we only need a goal function that we can optimize by selecting the parameters $$\Theta$$ and $$\Phi$$ correctly.

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/VAE_enc_dec1.svg" width="100%">
</div>

Neurons representing mean values and standard deviations usually do not have nonlinear activation functions.
As we have already stated, our desire is to maximize

$$
\log _{\mathbf{\theta }}p(\mathbf{x}^{(1)},\ldots
,\mathbf{x}^{(N)})=\sum _{i=1}^{N}\log
_{\mathbf{\theta }}p(\mathbf{x}^{(i)})
$$

With the appropriate transformation of $$\log(p(\mathbf x))$$, which is an element of the above sum, we get

$$
\text{}\log (p(\mathbf x))=D_{\mathit{KL}}\left(q( \mathbf z\vert \mathbf x)\parallel
p(\mathbf z\vert \mathbf x)\right)-D_{\mathit{KL}}\left(q(\mathbf z\vert \mathbf x)\parallel p(\mathbf z)\right)+\text{E}_{q(\mathbf z\vert \mathbf x)}\left(\log
(p(\mathbf x\vert \mathbf z))\right)
$$

$$D_{\mathit{KL}}$$ is [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) and represents the measure of the similarity of the two distributions. As we substitute $$p(\mathbf z \vert \mathbf x)$$ with $$q(\mathbf z \vert \mathbf x)$$, it is logical to try to make these two distributions as similar as possible. The KL divergence would then achieve the maximum, but as $$p(\mathbf z \vert \mathbf x)$$ is unknown, we maximize the remaining two parts. These two components together make the lower variational bound $$L$$ of $$log(p(\mathbf x))$$ and by maximizing the lower bound we increase the overall probability of the input sample $$\mathbf x$$.

$$
L(\mathbf{\theta ,\phi
,x^{(i)}})=-D_{\mathit{KL}}\left(q_{\mathbf{\phi
}}(\mathbf{z}\vert \mathbf{x}^{(i)})\parallel
p(\mathbf{z})\right)+\text{E}_{q_{\mathbf{\phi
}}(\mathbf{z}\vert \mathbf{x}^{(i)})}\left[\log
(p_{\mathbf{\theta
}}(\mathbf{x}^{(i)}\vert \mathbf{z}))\right]
$$

The second addend in the above equation can be seen as a success rate of reconstruction (the maximum is log(1) when the hidden layer $$\mathbf z$$ allows a perfect reconstruction). The first addend is considered a regularization component, and it supports the equalization of distributions $$q(\mathbf z \vert \mathbf x)$$ and $$p(\mathbf z)$$.

With selected approximations
$$q_{\mathbf{\phi }}(z\vert x)=N(\mu _{z}(x),\sigma _{z}(x))
$$
$$
p(z)=N(0,1)
$$
$$
p_{\mathbf{\theta }}(x\vert z)=N(\mu _{x}(z),\sigma _{x}(z))
$$
the two components of the lower variational bound become

$$\begin{equation*}
-D_{\mathit{KL}}\left(q_{\mathbf{\phi
}}(\mathbf{z}\vert \mathbf{x}^{(i)})\parallel
p(\mathbf{z})\right)=\frac{1}{2}\sum _{j}\left(1+\log (\sigma
_{z_{j}}^{(i)2})-\mu _{z_{j}}^{(i)2}-\sigma _{z_{j}}^{(i)2}\right)
\end{equation*}
$$

$$
\text{E}_{q_{\mathbf{\phi
}}(\mathbf{z}\vert \mathbf{x}^{(i)})}\left[\log
(p_{\mathbf{\theta
}}(\mathbf{x}^{(i)}\vert \mathbf{z}))\right]\approx
\frac{1}{K}\sum _{k=1}^{K}\log \left(p_{\mathbf{\theta
}}(\mathbf{x}^{(i)}\vert \mathbf{z}^{(i,k)})\right)\approx -\sum
_{j}{\frac{1}{2}\log (\sigma _{x_{j}}^{(i,k)2})+\frac{(x_{j}^{(i)}-\mu
_{x_{j}}^{(i,k)})^{2}}{2\sigma _{x_{j}}^{(i,k)2}}}
$$

Usually, $$K$$ is set to 1 to reduce the amount of sampling, provided that the size of the minibatch is at least 100.
The final expressions for the two components now give us the ultimate goal function for one input pattern. The average value for all input samples $$\mathbf x ^ {(i)}$$ is optimized! It is still necessary to slightly modify the structure of the network to allow backpropagation into the encoder layers. It is necessary to transform the stochastic neurons $$\mathbf z$$ into deterministic neurons with the stochastic attachment (the noise generator ε with the normal distribution $$N(0,1)$$).

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/VAE_enc_dec2.svg" width="100%">
</div>

Notice that the final structure of the network includes stochastic sampling, but the corresponding network parts do not affect the error gradient propagation. This also includes network outputs that, perhaps unexpectedly, do not participate in the goal function. In the final expression of the goal function, mean values and standard deviations of the output and hidden variables appear, which are actually the outputs of the encoder and the decoder. Standard deviation values are always positive, but network outputs usually aren't. To use the full range and reduce the number of required calculations, the network outputs are set to $$log(\sigma^2)$$ instead of $$σ$$.

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/VAE_enc_dec3.svg" width="100%">
</div>

The final VAE training algorithm is now:
1. Initialize parameters $$\Theta$$ i $$\Phi$$
2. Do
3. &nbsp; &nbsp; Choose random mini batch $$\mathbf X^M$$
4. &nbsp; &nbsp; Sample ε
5. &nbsp; &nbsp; Determine gradient of $$L$$ with respect to $$\Theta$$ and $$\Phi$$
6. &nbsp; &nbsp; Calculate new values for $$\Theta$$ and $$\Phi $$ according to the gradient
7. While $$\Theta$$ and $$\Phi$$ are not converging

So, by this procedure, we maximize the lower bound of the log probability of the input samples. This gives us the confidence that the log probability itself will grow, but there is no guarantee. Theoretically, it may happen that the lower bound is growing, and the probability itself is lowering, but in practice, it is most often not the case. The possible explanation for this effect lies in the fact that with a sufficiently complex encoder, $$q(\mathbf z \vert \mathbf x)$$ becomes complex and allows to approximate the distribution of $$p(\mathbf z \vert \mathbf x)$$, which maximizes the first (neglected) member of expression for $$log(p(\mathbf x))$$.

Generating new samples is performed only in the decoder part with the random initialization of the hidden layer $$\mathbf z$$ according to its distribution
$$
p(z)=N(0,1)
$$
or some specific vector $$\mathbf z$$.

In this task also, the MNIST database is used, whose images are treated as a series of imaginary binary pixels $$x_i$$ with Bernoulli's distribution and probability set by the value of input image pixels $$p(x_i = 1) = x_i^{in}$$. Then it is more appropriate for the decoder to implement Bernoulli's distribution instead of Gaussian. The output of the decoder can then represent the probability of $$p(x_i = 1)$$, which is also the expected value of output $$x_i$$. The probability itself can be defined as

$$
p(x_{i}^{\mathit{out}}=1)=\sigma \left(\sum
_{j=1}^{N}w_{\mathit{ji}}h_{j}+b_{i}\right)
$$

where $$\mathbf W$$ and $$\mathbf b$$ are weights and biasses linking the last layer of the decoder ($$\mathbf h$$) with the probability of the output variable $$x_i$$.
In accordance with this change, it is necessary to change the goal function, more precisely, its part relating to the reconstruction success rate

$$\text{E}_{q_{\mathbf{\phi
}}(\mathbf{z}\vert \mathbf{x}^{(i)})}\left[\log
(p_{\mathbf{\theta
}}(\mathbf{x}^{(i)}\vert \mathbf{z}))\right]$$

With binary variables with Bernoulli's distribution, the expression becomes

$$
\text{E}_{q_{\mathbf{\phi
}}(\mathbf{z}\vert \mathbf{x}^{(i)})}\left[\log
(p_{\mathbf{\theta
}}(\mathbf{x}^{(i)}\vert \mathbf{z}))\right]=-\sum
_{j}\left[x_{j}^{\text{in}}\log
p(x_{j}^{\mathit{out}}=1)+(1-x_{j}^{\text{in}})\log
(1-p(x_{j}^{\mathit{out}}=1))\right]=\sum
_{j}H\left(p(x_{j}=1),p(x_{j}^{\mathit{out}}=1)\right)
$$

$$H$$ is cross-entropy. Fortunately, Tensorflow offers a built-in function for $$H(\mathbf x, \sigma(\mathbf y))$$: `tf.nn.sigmoid_cross_entropy_with_logits(y, x)`. Please note that the first argument function is not $$p(x_j = 1)$$!

<a name='4zad'></a>

### Task 4

Implement VAE with 20 hidden variables $$z$$. Input data are MNIST numbers. The encoder and decoder have two hidden layers, each with 200 neurons with "soft plus" activation functions.

**Subtasks:**

 1. Visualize the reconstruction results for the first 20 MNIST samples.
 2. Visualize the distribution of mean values and standard deviations of the hidden variables $$z$$ for a sufficient number of input samples
 3. Visualize the layout of test samples in the 2D hidden variable space.
 4. Repeat experiments from the previous subtasks with only 2 elements in the hidden layer $$\mathbf z$$.

Use the following template:

**REMARK**: In addition to filling out the missing code, the template should be tailored as needed, and can be customized freely. So please **be especially careful with the claims that some of the code is not working for you!**

```python
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['image.cmap'] = 'jet'

mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)
n_samples = mnist.train.num_examples

learning_rate = 0.001
batch_size = 100

n_hidden_recog_1=200 # encoder layer 1
n_hidden_recog_2=200 # encoder layer 2
n_hidden_gener_1=200 # decoder layer 1
n_hidden_gener_2=200 # decoder layer 2
n_z=2 # number of hidden variables
n_input=784 # MNIST data input (img shape: 28*28)
in_shape = (28,28)

def get_canvas(Z, ind, nx, ny, in_shape, batch_size, sess):
    """Draw reconstruction layout in the 2D space of the hidden variables
     Z - hidden vectors arranged in the grid around the origin
     ind - indices for cutting Z to batch_size blocks to send to graph - problem with random generator
     nx - grid range in x axis - hidden variables z0
     ny - grid range in  y axis - hidden variable z1
     in_shape - dimensions of one reconstruction i.e. input thumbnails
     batch_size - the size of the minibatch used by the graph
     sess - session of the graph
    """
    # get reconstructions for visualiations
    X = np.empty((0,in_shape[0]*in_shape[1])) # empty array for concatenation 
    # split hidden vectors into minibatches of batch_size due to TF random generator limitation
    for batch in np.array_split(Z,ind):
        # fill up last batch to full batch_size if neccessary
        # this addition will not be visualized, but is here to avoid TF error
        if batch.shape[0] < batch_size:
            batch = np.concatenate((batch, np.zeros((batch_size-batch.shape[0], batch.shape[1]))), 0)
        # get batch_size reconstructions and add them to array of previous reconstructions
        X = np.vstack((X, sess.run(x_reconstr_mean_out, feed_dict={z: batch})))
    # make canvas with reconstruction tiles arranged by the hidden state coordinates of each reconstruction
    # this is achieved for all reconstructions by clever use of reshape, swapaxes and axis inversion
    return (X[0:nx*ny,:].reshape((nx*ny,in_shape[0],in_shape[1])).swapaxes(0,1)
            .reshape((in_shape[0],ny,nx*in_shape[1])).swapaxes(0,1)[::-1,:,:]
            .reshape((ny*in_shape[0],nx*in_shape[1])))

def draw_reconstructions(ins, outs, states, shape_in, shape_state):
    """Visualization of inputs and associated reconstructions and hidden layer states
     ins - input vectors
     outs - reconstructed vectors
     states - state vectors of hidden layer
     shape_in - dimensions of input images eg (28,28)
     shape_state - dimension for 2D status display (eg for 100 states (10,10)
    """
    plt.figure(figsize=(8, 12*4))
    for i in range(20):

        plt.subplot(20, 4, 4*i + 1)
        plt.imshow(ins[i].reshape(shape_in), vmin=0, vmax=1, interpolation="nearest")
        plt.title("Test input")
        plt.subplot(20, 4, 4*i + 2)
        plt.imshow(outs[i][0:784].reshape(shape_in), vmin=0, vmax=1, interpolation="nearest")
        plt.title("Reconstruction")
        plt.subplot(20, 4, 4*i + 3)
        plt.imshow(states[i][0:(shape_state[0] * shape_state[1])].reshape(shape_state),
                   vmin=-4, vmax=4, interpolation="nearest")
        plt.colorbar()
        plt.title("States")
    plt.tight_layout()
    
def plot_latent(inmat, labels):
    """Draw sample positions in 2D latent space
     inmat - matrix of latent states
     labels - class labels
    """
    plt.figure(figsize=(8, 6)) 
    plt.axis([-4, 4, -4, 4])
    plt.gca().set_autoscale_on(False)

    plt.scatter(inmat[:, 0], inmat[:, 1], c=np.argmax(labels, 1))
    plt.colorbar()
    plt.xlabel('z0')
    plt.ylabel('z1')

def save_latent_plot(name):
    """ Saving the current figure
     name - file name
    """
    plt.savefig(name)
    
def weight_variable(shape, name):
    """Initialize weights"""
    # http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
    return tf.get_variable(name, shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape):
    """Initialize biases"""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial)

def variable_summaries(var, name):
    """Collecting data for Tensorboard"""
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram(name, var)

def vae_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.softplus):
    """Creating a hidden layer"""
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        weights = weight_variable([input_dim, output_dim], layer_name + '/weights')
        variable_summaries(weights,'weights')
        tf.summary.tensor_summary('weightsT', weights)
        biases = bias_variable([output_dim])
        variable_summaries(biases, 'biases')
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
    return activations

tf.reset_default_graph() 
    
sess = tf.InteractiveSession()
        
# Define input tensor
x = 

# Define encoder
layer_e1 = vae_layer(x, n_input, n_hidden_recog_1, 'layer_e1') 
layer_e2 = 

with tf.name_scope('z'):
# Define hidden variables and the associated noise generator
    z_mean = vae_layer(layer_e2, n_hidden_recog_2, n_z, 'z_mean', act=tf.identity)
    z_log_sigma_sq = 
    eps = tf.random_normal((batch_size, n_z), 0, 1, dtype=tf.float32)
                         
    z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))
    tf.summary.histogram('activations', z)

# Define decoder
layer_d1 = vae_layer(z, n_z, n_hidden_gener_1, 'layer_d1') 
layer_d2 = 
            
# Define the mean value of the reconstruction
x_reconstr_mean = 

x_reconstr_mean_out = tf.nn.sigmoid(x_reconstr_mean)

# Define two components of the cost function
with tf.name_scope('cost'):
    cost1 = 
    tf.summary.histogram('cross_entropy', cost1)
    cost2 = 
    tf.summary.histogram('D_KL', cost2)
    cost = tf.reduce_mean(tf.reduce_sum(cost1,1) + tf.reduce_sum(cost2,1))   # average over batch
    tf.summary.histogram('cost', cost)
                         
# ADAM optimizer
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Collecting data for Tensorboard
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()

saver = tf.train.Saver()

n_epochs = 100
train_writer = tf.summary.FileWriter('train', sess.graph)

sess.run(init)

total_batch = int(n_samples / batch_size)
step = 0
for epoch in range(n_epochs):
    avg_cost = 0.
        
    for i in range(total_batch):
        batch_xs, _ = mnist.train.next_batch(batch_size)
        # Fit training using batch data
        opt, cos = sess.run((optimizer, cost), feed_dict={x: batch_xs})
        # Compute average loss
        avg_cost += cos / n_samples * batch_size
        
    # Display logs per epoch step
    if epoch%(int(n_epochs/10)) == 0:
        print("Epoch:", '%04d' % (epoch+1),
              "cost=", "{:.9f}".format(avg_cost)) 
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, optimizer], feed_dict={x: batch_xs}, 
                              options=run_options, run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'epoch%03d' % epoch)
        train_writer.add_summary(summary, i)
        
        saver.save(sess, os.path.join('train', "model.ckpt"), epoch)

train_writer.close()

# visualization of reconstruction and states
x_sample = mnist.test.next_batch(100)[0]
x_reconstruct, z_out = sess.run([x_reconstr_mean_out, z], feed_dict={x: x_sample})

draw_reconstructions(x_sample, x_reconstruct, z_out, (28, 28), (4,5)) # adjust dimensions as needed

# Visualization of test samples in 2D hidden variable space - 1st variant
x_sample, y_sample = mnist.test.next_batch(5000)
z_mu, z_sigma = sess.run((z_mean, z_log_sigma_sq), feed_dict={x: x_sample})
    
plot_latent(z_mu, y_sample)
#save_latent_plot('trt.png')

# Visualization of test samples in 2D hidden variable space - 2nd variant

nx = ny = 21
x_values = np.linspace(-3, 3, nx)
y_values = np.linspace(-3, 3, ny)

canvas = np.empty((28*ny, 28*nx))

# Carefull filling of grid due to the fixed size of z batch in the graph
Xi, Yi = np.meshgrid(x_values, y_values)
Z = np.column_stack((Xi.flatten(), Yi.flatten()))
X = np.empty((0,28*28))
ind = list(range(batch_size, nx*ny, batch_size))
for i in np.array_split(Z,ind):
    if i.shape[0] < batch_size:
        i = np.concatenate((i, np.zeros((batch_size-i.shape[0], i.shape[1]))), 0)
    X = np.vstack((X, sess.run(x_reconstr_mean_out, feed_dict={z: i})))
    
for i, yi in enumerate(y_values):
    for j, xi in enumerate(x_values):
        canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = X[i*nx+j].reshape(28, 28)

plt.figure(figsize=(8, 10))
plt.imshow(canvas, origin="upper")
plt.xticks( np.linspace(14,588-14,11), np.round(np.linspace(-3,3,11), 2) )
plt.yticks( np.linspace(14,588-14,11), np.round(np.linspace(3,-3,11), 2) )
plt.xlabel('z0')
plt.ylabel('z1')
plt.tight_layout()

# Visualization of muted hidden layer elements - 1st variant

# Auxiliary function for drawing boxplot graphs
def boxplot_vis(pos, input_data, label_x, label_y):
    ax = fig.add_subplot(130+pos)
    plt.boxplot(input_data, 0, '', 0, 0.75)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    return ax
   
fig = plt.figure(figsize=(15,4))

# Visualization of statistics for z_mean
boxplot_vis(1,z_mu, 'Z mean values', 'Z elemets')

# Visualization of statistics for z_sigma
ax = boxplot_vis(2, np.square(np.exp(z_sigma)), 'Z sigma values', 'Z elemets')
ax.set_xlim([-0.05,1.1])

# Visualization of statistics for input decoder weights
test = tf.get_default_graph().get_tensor_by_name("layer_d1/weights:0")
weights_d1 = test.eval(session=sess)
boxplot_vis(3, weights_d1.T, 'Weights to decoder', 'Z elemets')


# Visualization of muted hidden layer elements - 1st variant

from mpl_toolkits.mplot3d import Axes3D

# Function to draw 3D bar graph
def bargraph_vis(pos, input_data, dims, color, labels):
    ax = fig.add_subplot(120+pos, projection='3d')
    xpos, ypos = np.meshgrid(range(dims[0]), range(dims[1]))
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)
    
    dx = np.ones_like(zpos) 
    dy = np.ones_like(zpos) * 0.5
    dz = input_data.flatten()
    
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=color)
    ax.view_init(elev=30., azim=5)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
                             
fig = plt.figure(figsize=(15,7))

# 3D bar graph for z_mean
labels = ('Samples', 'Hidden elements', 'Z mean')
bargraph_vis(1, z_mu, [200, z_mu.shape[1]], 'g', labels)

# 33D bar graph for weights connecting z_mena and decoder
labels = ('Decoder elements', 'Hidden elements Z', 'Weights')
bargraph_vis(2, weights_d1.T, weights_d1.T.shape, 'y', labels)


```

#### Bonus task - Tensorboard

The template for Task 4 contains the data collection code that can be displayed using [Tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard). Run Tensorboard and check what information is available on a trained VAE.

<a name='gan'></a>

### Generative asdversarial networks (GAN)

GAN's primary purpose is to generate new and persuasive samples, but the working principle is slightly different from the previous two models. GAN does not directly evaluate the parameters of $$p(\mathbf x)$$ or any other distribution, although its training can be interpreted as an estimate of $$p(\mathbf x)$$. Most likely due to this different approach, GANs often generate visually the best samples compared to VAE or other generative networks.

GAN consists of two separate networks:

1. Generator (G) whose task is to generate convincing samples
2. A discriminator (D) whose task is to identify whether a sample is a genuine (from a training set) or an artificial sample generated by G

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/GAN.svg" width="100%">
</div>

These two networks are adversaries as they have diametrically opposed goals and are trying to outsmart each other. This competition makes them better in achieving their own goal and puts their focus on all the essential details of input data. Eventually, their competing should result in the generator that generates perfect samples that the discriminator can't distinguish from the training samples. In order for the generator to achieve that, it is necessary for the discriminator to be very successful in its own task.

Generator outputs samples for some random input vector at its input which obeys some preselected distribution. This randomness at the input allows the generator to always generate new, unseen samples. There are no special limitations on the architecture of the generator, but it is desirable to be trainable using the backpropagation algorithm.

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/G.svg" width="50%">
</div>

At its output, the discriminator should estimate the class of an input sample, genuine or artificial. Unlike a generator, it is possible to use supervised learning because the class of each sample is known. For simplicity, the output of the discriminator can be limited to $$ [0,1] $$ and interpret as a probability that the input sample is real (from the training set).

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/D.svg" width="50%">
</div>

The described discriminator and generator goals can be formally expressed in the following goal function:

$$\min_G \max_D V(D,G) = E_{ \mathbf x \sim p_{data}(\mathbf x) } [\log D( \mathbf x)] + E_{ \mathbf z  \sim p_{\mathbf z}(\mathbf z) } [\log(1 - D(G( \mathbf z)))]$$

The first addend represents the expectation of estimated log probability that the samples from the training set are genuine. The second addend represents the expectation of a log probability estimation that the generated samples are not real, ie. generated. The discriminator aims to maximize both addends, while the generator aims to minimize just the second addend. Each addend can be easily evaluated for a mini-batch and gradients can be estimated for parameters of both networks

Training of both networks (G and D) can be carried out simultaneously or in one iteration one can first train one network and then the other. In addition, some authors recommend that one network is trained for several iterations and then the other network for just one iteration. Appropriate goal functions of both networks can be implemented with the imaginative use of the [`tf.nn.sigmoid_cross_entropy_with_logits (y, x)`](https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits) function.

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/GAN2.svg" width="100%">
</div>

As a discriminator needs to receive input samples from two different sources, we can implement it in the TensorFlow environment as two identical networks each receiving input samples from one source, with both discriminators sharing or using the same weights. Training each individual discriminator then causes modifications of the same set of weights. Sharing network parameters or variables in the TensorFlow environment is achieved by using the variable scope [`tf.variable_scope`](https://www.tensorflow.org/api_docs/python/tf/variable_scope) and the `reuse` keyword.

Deep Convolutional GAN ​​(DCGAN) provide very good results for image generation, and they use convolutional layers as hidden layers in both the generator and the discriminator. Unlike conventional convolutional networks, pooling layers are not used here, but subsampling is performed using convolution layers with stride greater than 1. The authors recommend using Batch normalization in all layers except in the output layer of the generator, and the input and output layers of the discriminator. The use of Leaky ReLU activation functions in all layers except in outputs is another specificity of DCGAN as well as the elimination of fully-connected layers.

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/DCGAN.svg" width="100%">
</div>

<a name='5zad'> </a>

### Task 5

Implement DCGAN with the generator (4 convolution layers) and a discriminator (3 convolution layers). Use kernel size [4,4] in all convolutions except for the output layer of the discriminator. The number of channels from the input to the output layers should be G: 512, 256, 128, 1 and D: 64, 128, Generator input $$\mathbf z$$ should have 100 elements obeying the normal distribution $$ N (0,1) $$. Use MNIST numbers scaled to size 32x32 as training set and train the network for at least 20 epochs. In each iteration, perform optimization of the generator and one optimization of the discriminator with one mini-batch each. Use a tanh activation function for the generator output and sigmoid activation for the discriminator output.

**Subtasks:**

 1. Visualize the results of generating 100 new samples from random variables $$ \ mathbf z $$. Compare the results with samples generated by VAE.
 2. In one iteration, use two mini-batches to train the generator and only one mini-batch to train the discriminator. Visualize generated samples. Repeat the same procedure with two mini-batches for the discriminator and one for the generator. Comment on the results.
 3. Turn off batch normalization in both networks. Comment on the results.

Use the following template:

**REMARK**: In addition to filling out the missing code, the template should be tailored as needed, and can be customized freely. So please **be especially careful with the claims that some of the code is not working for you!**

```python
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils import tile_raster_images
import matplotlib.pyplot as plt
import math

%matplotlib inline
plt.rcParams['image.cmap'] = 'jet'

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])
n_samples = mnist.train.num_examples

# training parameters
batch_size = 100
lr = 0.0002
n_epochs = 20

def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

# D(x)
def discriminator(x, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # 1st hidden layer
        conv = tf.layers.conv2d(x, 64, [4, 4], strides=(2, 2), padding='same')
        lrelu_ = lrelu(conv, 0.2)
        
        # 2nd hidden layer
        
        
        # output layer
        
        
        return out, conv

# G(z)
def generator(z, isTrain=True):
    with tf.variable_scope('generator'):
        # 1st hidden layer
        conv = tf.layers.conv2d_transpose(z, 512, [4, 4], strides=(1, 1), padding='valid')
        lrelu_ = lrelu(tf.layers.batch_normalization(conv, training=isTrain))
        
        # 2nd hidden layer
        
        # 3rd hidden layer
        
        # output layer

        
        return out

def show_generated(G, N, shape=(32,32), stat_shape=(10,10), interpolation="bilinear"):
    """Visualization of generated samples
     G - generated samples
     N - number of samples
     shape - dimensions of samples eg (32,32)
     stat_shape - dimension for 2D sample display (eg for 100 samples (10,10)
    """
    
    image = (tile_raster_images(
        X=G,
        img_shape=shape,
        tile_shape=(int(math.ceil(N/stat_shape[0])), stat_shape[0]),
        tile_spacing=(1, 1)))
    plt.figure(figsize=(10, 14))
    plt.imshow(image, interpolation=interpolation)
    plt.axis('off')
    plt.show()
    

def gen_z(N, batch_size):
    z = np.random.normal(0, 1, (batch_size, 1, 1, N))
    return z

# input variables
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))
isTrain = tf.placeholder(dtype=tf.bool)
    
# generator
G_z = generator(z, isTrain)
    
# discriminator
# real
D_real, D_real_logits = discriminator(x, isTrain)
# fake
...


# labels for learning
true_labels = tf.ones([batch_size, 1, 1, 1]
true_labels = tf.zeros([batch_size, 1, 1, 1]
# loss for each network                       
D_loss_real =
D_loss_fake =
D_loss = D_loss_real + D_loss_fake
G_loss = 

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.3).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.3).minimize(G_loss, var_list=G_vars)


# open session and initialize all variables
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)
tf.global_variables_initializer().run()

# MNIST resize and normalization
train_set = tf.image.resize_images(mnist.train.images, [32, 32]).eval()
# input normalization
...

#fixed_z_ = np.random.uniform(-1, 1, (100, 1, 1, 100))
fixed_z_ = gen_z(100, 100)
total_batch = int(n_samples / batch_size)

for epoch in range(n_epochs):
    for iter in range(total_batch):
        # update discriminator
        x_ = train_set[iter*batch_size:(iter+1)*batch_size]
        
        # update discriminator
        
        z_ = gen_z(100, batch_size)
        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, isTrain: True})
                

        # update generator
        ...
            
    print('[%d/%d] loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), n_epochs, loss_d_, loss_g_))
    
    test_images = sess.run(G_z, {z: fixed_z_, isTrain: False})
    show_generated(test_images, 100)
```

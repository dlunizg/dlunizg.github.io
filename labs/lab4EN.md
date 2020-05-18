---
layout: page
mathjax: true
permalink: /lab4en/
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
_v1.3-2020_

NOTE: If the execution time for some assignments is taking too long due to a lack of access to GPU compute resources, we want to point out that Google offers Google Colab as a free service which offers GPU compute for free.

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
_{j=1}^{N}w_{\mathit{ji}}h_{j}+a_{i}\right)$$ za vidljivi Layer

$$p(h_{j}=1)=\sigma \left(\sum
_{i=1}^{N}w_{\mathit{ji}}v_{i}+b_{j}\right)$$ za skriveni Layer

The sampling of the values of a particular variable is carried out according to the above two equations and using a random number generator.


```python
sampled_tensor = probability_tensor.bernoulli()
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


**REMARK**: In addition to filling out the missing code, the template should be tailored as needed, and can be customized freely. So please **be especially careful with the claims that some of the code is not working for you!**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

from torchvision import datasets, transforms
import tqdm
from torchvision.utils import make_grid

import torch.distributions as tdist

import numpy as np
import tqdm

import matplotlib.pyplot as plt

BATCH_SIZE = 100
EPOCHS = 100
VISIBLE_SIZE = 784
HIDDEN_SIZE = 100

def visualize_RBM_weights(weights, grid_width, grid_height, slice_shape=(28, 28)):
    for idx in range(0, grid_width * grid_height):
        plt.subplot(grid_height, grid_width, idx+1)
        plt.imshow(weights[..., idx].reshape(slice_shape))
        plt.axis('off')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def draw_rec(inp, title, size, Nrows, in_a_row, j):
    plt.subplot(Nrows, in_a_row, j)
    plt.imshow(inp.reshape(size), vmin=0, vmax=1, interpolation="nearest")
    plt.title(title)
    plt.axis('off')
    
    
def reconstruct(ind, states, orig, weights, biases, h1_shape=(10, 10), v_shape=(28,28)):
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
    
    for i in range(h1_shape[0] * h1_shape[1]):
        if states[ind,i] > 0:
            j += 1
            reconstr = reconstr + weights[:,i]
            titl = '+= s' + str(i+1)
            draw_rec(sigmoid(reconstr), titl, v_shape, Nrows, in_a_row, j)
    plt.tight_layout()

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor()
                               ])), batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor()
                               ])), batch_size=BATCH_SIZE)

class RBM():
    
    def __init__(self, visible_size, hidden_size, cd_k=1):
        self.v_size = visible_size
        self.h_size = hidden_size
        self.cd_k = cd_k
        
        normal_dist = tdist.Normal(0, 0.1)
        
        self.W = torch.Tensor(normal_dist.sample(sample_shape=(self.v_size, self.h_size)))
        self.v_bias = torch.Tensor(torch.zeros(self.v_size))
        self.h_bias = torch.Tensor(torch.zeros(self.h_size))

    
    def forward(self, batch):
        return self._cd_pass(batch)
    
    
    def __call__(self, batch):
        return self.forward(batch)
    
    
    def _cd_pass(self, batch):
        batch = batch.view(-1, 784)
        h0_prob = 
        h0 = 

        h1 = h0

        for step in range(0, self.cd_k):
            v1_prob = 
            v1 = 
            h1_prob = 
            h1 = 
            
        return h0_prob, h0, h1_prob, h1, v1_prob, v1
    
    def reconstruct(self, h, gibbs_steps=None):
        h1 = h
        
        steps_to_do = self.cd_k
        if gibbs_steps is not None:
            steps_to_do = gibbs_steps

        for step in range(0, steps_to_do):
            v1_prob = 
            v1 = 
            h1_prob = 
            h1 = 

        return h1_prob, h1, v1_prob, v1

    
    def update_weights_for_batch(self, batch, learning_rate=0.01):
        h0_prob, h0, h1_prob, h1, v1_prob, v1 = self._cd_pass(batch)

        w_positive_grad = 
        w_negative_grad = 

        dw = (w_positive_grad - w_negative_grad) / batch.shape[0]

        self.W = self.W + 
        self.v_bias = self.v_bias + 
        self.h_bias = self.h_bias + 


model = RBM(visible_size=VISIBLE_SIZE, hidden_size=HIDDEN_SIZE, cd_k=1)
for curr_epoch in tqdm.tqdm(range(0, EPOCHS)):
    for sample, label in train_loader:
        sample = sample.view(-1, 784)
        model.update_weights_for_batch(sample, 0.1)


plt.figure(figsize=(12, 12), facecolor='w')
visualize_RBM_weights(model.W.data, 10, 10)


sample, _ = next(iter(test_loader))
sample = sample.view(-1, 784)

for idx in range(0, 20): 
    h0_prob, h0, h1_prob, h1, v1_prob, v1 = model(sample)


    plt.figure(figsize=(8, 4), facecolor='w')
    plt.subplot(1, 3, 1)
    plt.imshow(sample[idx, ...].view(28, 28).cpu())
    if idx == 0:
        plt.title("Original image")

    plt.subplot(1, 3, 2)
    recon_image = v1_prob[idx, ...].view(28, 28)
    plt.imshow(recon_image.cpu().data)
    if idx == 0:
        plt.title("Reconstruction")
    
    plt.subplot(1, 3, 3)
    state_image = h1[idx, ...].view(10, 10)
    plt.imshow(state_image.cpu().data)
    if idx == 0:
        plt.title("Hidden state")

sample, _ = next(iter(test_loader))
sample = sample[0, ...].view(-1, 784)

h0_prob, h0, h1_prob, h1, v1_prob, v1 = model(sample)

reconstruct(0, h1.numpy(), sample.numpy(), model.W.numpy(), model.v_bias.numpy())


sample, _ = next(iter(test_loader))
sample = sample.view(-1, 784)

h0_prob, h0, h1_prob, h1, v1_prob, v1 = model(sample)

h0_prob, h0, h1_prob, h1, v1_prob, v1, model_weights, model_v_biases = list(map(lambda x: x.numpy(), [h0_prob, h0, h1_prob, h1, v1_prob, v1, model.W, model.v_bias]))



plt.figure(figsize=(9, 4))
tmp = (h1.sum(0)/h1.shape[0]).reshape((10, 10))
plt.imshow(tmp, vmin=0, vmax=1, interpolation="nearest")
plt.axis('off')
plt.colorbar()
plt.title('Probability of activation per neuron of the hidden layer')


plt.figure(figsize=(16, 16))
tmp_ind = (-tmp).argsort(None)
visualize_RBM_weights(model_weights[:, tmp_ind], 10, 10)
plt.suptitle('Sorted weight matrices')


r_input = np.random.rand(100, HIDDEN_SIZE)
r_input[r_input > 0.9] = 1 
r_input[r_input < 1] = 0
r_input = r_input * 20 

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

h1_prob, h1, v1_prob, v1 = model.reconstruct(torch.from_numpy(r_input).float(), 19)

plt.figure(figsize=(16, 16))
for idx in range(0, 19):
    plt.figure(figsize=(14, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(r_input[idx, ...].reshape(10, 10))
    if idx == 0:
        plt.title("Set state")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(h1[idx, ...].view(10, 10))
    if idx == 0:
        plt.title("Final state")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(v1_prob[idx, ...].view(28, 28))
    if idx == 0:
        plt.title("Reconstruction")
    plt.axis('off')
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
class DBN():

    def __init__(self, first_rbm: RBM, second_hidden_size, cd_k=1):
        self.v_size = first_rbm.v_size
        self.h1_size = first_rbm.h_size
        self.h2_size = second_hidden_size
        self.cd_k = cd_k
        
        normal_dist = tdist.Normal(0, 0.1)
        
        self.W1 = first_rbm.W
        self.v_bias = first_rbm.v_bias.clone()
        self.h1_bias = first_rbm.h_bias.clone()
        
        self.W2 = torch.Tensor(normal_dist.sample(sample_shape=(self.h1_size, self.h2_size)))
        self.h2_bias = torch.Tensor(torch.zeros(self.h2_size))
    
    
    def forward(self, batch, steps=None):
        batch = batch.view(-1, 784)
        
        h1up_prob = 
        h1up = 
        
        h2up_prob = 
        h2up = 
        
        h1down_prob, h1down, h2down_prob, h2down = self.gibbs_sampling(???, steps)
        
        return h1up_prob, h1up, h2up_prob, h2up, h1down_prob, h1down, h2down_prob, h2down

    
    def gibbs_sampling(self, h2, steps=None):
        h2down = h2
        
        steps_to_do = self.cd_k
        
        if steps is not None:
            steps_to_do = steps

        for step in range(0, steps_to_do):
            h1down_prob = 
            h1down = 
            
            h2down_prob = 
            h2down = 
            
        return h1down_prob, h1down, h2down_prob, h2down 
    
    def reconstruct(self, h2, steps=None):
        _, _, h2down_prob, h2down = self.gibbs_sampling(???, steps)
        
        h1down_prob = 
        h1down = 
        
        v_prob = 
        v_out = 
        
        return v_prob, v_out, h2down_prob, h2down
    
    def update_weights_for_batch(self, batch, learning_rate=0.01):
        h1up_prob, h1up, h2up_prob, h2up, h1down_prob, h1down, h2down_prob, h2down = self.forward(batch)

        w2_positive_grad = 
        w2_negative_grad = 

        dw2 = (w2_positive_grad - w2_negative_grad) / h1up.shape[0]

        self.W2 = self.W2 + 
        self.h1_bias = self.h1_bias + 
        self.h2_bias = self.h2_bias + 
        
                
    
    def __call__(self, batch):
        return self.forward(batch)


dbnmodel = DBN(model, second_hidden_size=100, cd_k=2)
for curr_epoch in tqdm.tqdm(range(0, EPOCHS)):
    for sample, label in train_loader:
        sample = sample.view(-1, 784)
        dbnmodel.update_weights_for_batch(sample, learning_rate=0.1)

plt.figure(figsize=(12, 12), facecolor='w')
visualize_RBM_weights(dbnmodel.W2.data.cpu(), 10, 10, slice_shape=(10, 10))

sample, _ = next(iter(test_loader))
sample = sample.view(-1, 784)

for idx in range(0, 20):
    h1up_prob, h1up, h2up_prob, h2up, h1down_prob, h1down, h2down_prob, h2down = dbnmodel(sample[idx, ...])
    v_prob, v, _, _ = dbnmodel.reconstruct(h2down)

    plt.figure(figsize=(4*3, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(sample[idx,...].view(28, 28))
    if idx == 0:
        plt.title("Test input")
    
    plt.subplot(1, 3, 2)
    plt.imshow(v_prob[0, ...].view(28, 28))
    if idx == 0:
        plt.title("Reconstruction")
    
    plt.subplot(1, 3, 3)
    plt.imshow(h2down.view(10, 10))
    if idx == 0:
        plt.title("Hidden state")


r_input = np.random.rand(100, HIDDEN_SIZE)
r_input[r_input > 0.9] = 1 # postotak aktivnih - slobodno varirajte
r_input[r_input < 1] = 0
r_input = r_input * 20 # pojačanje za slučaj ako je mali postotak aktivnih

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

v_out_prob, v_out, h2down_prob, h2down = dbnmodel.reconstruct(torch.from_numpy(r_input).float(), 100)

plt.figure(figsize=(16, 16))
for idx in range(0, 19):
    plt.figure(figsize=(14, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(r_input[idx, ...].reshape(10, 10))
    if idx == 0:
        plt.title("Set state")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(h2down[idx, ...].view(10, 10))
    if idx == 0:
        plt.title("Final state")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(v_out_prob[idx, ...].view(28, 28))
    if idx == 0:
        plt.title("Reconstruction")
    plt.axis('off')
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
class DBNWithFineTuning():

    def __init__(self, base_dbn: DBN, cd_k=1):
        self.v_size = base_dbn.v_size
        self.h1_size = base_dbn.h1_size
        self.h2_size = base_dbn.h2_size
        self.cd_k = cd_k
        
        normal_dist = tdist.Normal(0, 0.1)
        
        self.R1 = base_dbn.W1.clone()
        self.W1_down = base_dbn.W1.T.clone()
        self.v1_bias = base_dbn.v_bias.clone()
        self.h1_up_bias = base_dbn.h1_bias.clone()
        self.h1_down_bias = base_dbn.h1_bias.clone()
        
        self.W2 = base_dbn.W2.clone()
        self.h2_bias = base_dbn.h2_bias.clone()
    
    
    def forward(self, batch, steps=None):
        batch = batch.view(-1, 784)
        
        h1_up_prob = 
        h1_up = 
        
        v1_up_down_prob = 
        v1_up_down = 
        
        h2_up_prob = 
        h2_up = 
        
        h1_down_prob, h1_down, h2_down_prob, h2_down = self.gibbs_sampling(???, steps=steps)
        
        v1_down_prob = 
        v1_down = 
        
        h1_down_up_prob = 
        h1_down_up = 
        
        return h1_up_prob, h1_up, v1_up_down_prob, v1_up_down, h2_up_prob, h2_up, h1_down_prob, h1_down, h2_down_prob, h2_down, v1_down_prob, v1_down, h1_down_up_prob, h1_down_up
    
    def gibbs_sampling(self, h2, steps=None):
        h2_down = h2
        
        steps_to_do = self.cd_k
        
        if steps is not None:
            steps_to_do = steps
        
        
        for step in range(0, self.cd_k):
            h1_down_prob =
            h1_down = 

            h2_down_prob = 
            h2_down = 
            
        return h1_down_prob, h1_down, h2_down_prob, h2_down


    
    def reconstruct(self, h2, steps=None):
        h1_down_prob, h1_down, h2_down_prob, h2down = self.gibbs_sampling(???, steps)
        
        v_out_tmp_prob = 
        v_out_tmp =
        v_out_prob = 
        v_out = 
        
        return v_out_prob, v_out, h2_down_prob, h2down
    
    def update_weights_for_batch(self, batch, learning_rate=0.01):
        h1_up_prob, h1_up, v1_up_down_prob, v1_up_down, h2_up_prob, h2_up, h1_down_prob, h1_down, h2_down_prob, h2_down, v1_down_prob, v1_down, h1_down_up_prob, h1_down_up = self.forward(batch)
        
        self.W1_down = self.W1_down + 
        self.R1 = self.R1 + 
        
        self.v1_bias = self.v1_bias +
        
        self.h1_down_bias = self.h1_down_bias + 
        self.h1_up_bias = self.h1_up_bias + 
        
        
        w2_positive_grad = 
        w2_negative_grad = 
        dw2 = (w2_positive_grad - w2_negative_grad) / h1_up.shape[0]
        
        self.W2 = self.W2 + 
        self.h2_bias = self.h2_bias + 
        
    
    def __call__(self, batch):
        return self.forward(batch)


dbnmodel_ft = DBNWithFineTuning(dbnmodel, cd_k=2)
for curr_epoch in tqdm.tqdm(range(0, EPOCHS)):
    for sample, label in train_loader:
        sample = sample.view(-1, 784)
        dbnmodel_ft.update_weights_for_batch(sample, 0.01)

plt.figure(figsize=(12, 12), facecolor='w')
visualize_RBM_weights(dbnmodel_ft.R1.data, 10, 10)
plt.tight_layout()


plt.figure(figsize=(12, 12), facecolor='w')
visualize_RBM_weights(dbnmodel_ft.W1_down.T.data, 10, 10)
plt.tight_layout()

difference = torch.abs(dbnmodel_ft.R1.data - dbnmodel_ft.W1_down.T.data)
plt.figure(figsize=(12, 12), facecolor='w')
visualize_RBM_weights(difference, 10, 10)
plt.tight_layout()

sample, _ = next(iter(test_loader))
sample = sample.view(-1, 784)

for idx in range(0, 20): 
    # rbn reconstruct
    _, _, _, _, recon1, _ = model(sample[idx, ...])
    
    # dbn reconstruct
    _, _, _, _, _, _, _, h2down = dbnmodel.forward(sample[idx, ...])
    recon2, _, _, _ = dbnmodel.reconstruct(h2down)
    
    # dbn fine tune reconstruct
    _, _, _, _, _, _, _, _, _, h2_down, _, _, _, _ = dbnmodel_ft(sample[idx, ...])
    recon3, _, _, _ = dbnmodel_ft.reconstruct(h2_down, 2)
    
    plt.figure(figsize=(5*3, 3))
    plt.subplot(1, 5, 1)
    plt.imshow(sample[idx, ...].view(28, 28))
    if idx == 0:
        plt.title("Original image")
    
    plt.subplot(1, 5, 2)
    plt.imshow(recon1.view(28, 28))
    if idx == 0:
        plt.title("Reconstruction 1")
    
    plt.subplot(1, 5, 3)
    plt.imshow(recon2.view(28, 28))
    if idx == 0:
        plt.title("Reconstruction 2")
    
    plt.subplot(1, 5, 4)
    plt.imshow(recon3.view(28, 28))
    if idx == 0:
        plt.title("Reconstruction 3")
    
    plt.subplot(1, 5, 5)
    plt.imshow(h2_down.view(10, 10))
    if idx == 0:
        plt.title("Top state 3")

r_input = np.random.rand(100, 100)
r_input[r_input > 0.9] = 1
r_input[r_input < 1] = 0

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

    
v_out_prob, v_out, h2_down_prob, h2down = dbnmodel_ft.reconstruct(torch.from_numpy(r_input).float(), 100)

plt.figure(figsize=(16, 16))
for idx in range(0, 19):
    plt.figure(figsize=(14, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(r_input[idx, ...].reshape(10, 10))
    if idx == 0:
        plt.title("Set state")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(h2down[idx, ...].view(10, 10))
    if idx == 0:
        plt.title("Final state")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(v_out_prob[idx, ...].view(28, 28))
    if idx == 0:
        plt.title("Reconstruction")
    plt.axis('off')
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

from torchvision import datasets, transforms
import tqdm
from torchvision.utils import make_grid

import torch.distributions as tdist

import numpy as np
import tqdm

import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

class VAE(nn.Module):
    
    def __init__(self, latent_size):
        super(VAE, self).__init__()
        
        self.latent_size = latent_size
        
        ???
        
    def encode(self, x):
        # TODO!
        return ???, ???
        
    def reparametrize(self, mu, logvar):
        # TODO!
        return ???
    
    def decode(self, z):
        # TODO!
        return ???
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparametrize(mu, logvar)
        reconstruction = self.decode(z)
        
        return reconstruction, z, mu, logvar
    
    @staticmethod
    def loss_fn(reconstruction, batch, mu, logvar):
        crossentropy = 
        kl_div = 
        return ???

LATENT_SIZE = 2
model = VAE(LATENT_SIZE)
model = train(model, batch_size=1024, device='cuda', n_epochs=100, log_epochs=10, learning_rate=3.24e-4)

plot_reconstructions('cuda', state_shape=(2, 1))

_, test_loader = prepare_data_loaders()
df_mu, df_logvar, df_dec1_weights = generate_latent_dataframes(test_loader)
plt.figure(figsize=(16, 16))
sns.scatterplot(x='mu_z0', y='mu_z1', hue='label', s=50, data=df_mu)

plot_data_boxplots(df_mu, df_logvar, df_dec1_weights)

walk_in_latent_space(latent_space_abs_limit=1.5, sqrt_sample_count=15, latent_size=LATENT_SIZE, dimensions_to_walk=(0,1))
```

The above used functions can be done by the students, or you may use the implementations provided below:

```python
def prepare_data_loaders(batch_size=32):
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor()
                               ])), batch_size=batch_size)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./files', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor()
                                   ])), batch_size=batch_size)
    
    return train_loader, test_loader

def train(model, n_epochs=10, log_epochs=1, batch_size=32, learning_rate=1e-3, device='cpu'):
    train_loader, test_loader = prepare_data_loaders(batch_size)
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    
    for epoch_idx in range(0, n_epochs):
        
        train_loss = 0
        for batch_idx, (image_data, _) in enumerate(train_loader):
            image_data = image_data.to(device)
            
            optimizer.zero_grad()
            reconstructed_batch, batch_z, batch_mu, batch_logvar = model(image_data)
            loss = model.loss_fn(reconstructed_batch, image_data, batch_mu, batch_logvar)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        if epoch_idx % log_epochs == 0:
            print(f"Epoch {epoch_idx+1}/{n_epochs}: {train_loss / (len(train_loader) * train_loader.batch_size):.2f}")
            
    model.eval()
    
    return model

def plot_reconstructions(device='cpu', number_of_samples=10, state_shape=(4, 5)):
    train_loader, test_loader = prepare_data_loaders(batch_size=number_of_samples)
    batch, _ = next(iter(test_loader))
    recons, zs, mus, logvars = model(batch.to(device))
    
    for idx in range(0, number_of_samples):
        original_image = batch[idx, ...].view(28, 28).data.cpu()
        recon_image = recons[idx, ...].view(28, 28).data.cpu()
        state = zs[idx, ...].view(*state_shape).data.cpu()

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)

        plt.subplot(1, 3, 2)
        plt.imshow(recon_image)
        
        plt.subplot(1, 3, 3)
        plt.imshow(state)
        plt.clim(-4, 4)
        plt.colorbar()

def generate_latent_dataframes(data_loader):
    mu_acc = []
    logvar_acc = []
    label_acc = []

    for image_data, label in tqdm.tqdm(data_loader):
        mu, logvar = model.encode(image_data.view(-1, 784).to('cuda'))

        mu_acc.extend(mu.data.cpu().numpy())
        logvar_acc.extend(logvar.data.cpu().numpy())
        label_acc.extend(label.data.cpu().numpy())

    mu_acc = np.array(mu_acc)
    logvar_acc = np.array(logvar_acc)


    tmp = {
        'label': label_acc
    }
    for idx in range(0, mu_acc.shape[1]):
        tmp[f'mu_z{idx}'] = mu_acc[..., idx]

    df_mu = pd.DataFrame(tmp)
    df_mu['label'] = df_mu['label'].astype('category')


    tmp = {
        'label': label_acc
    }
    for idx in range(0, mu_acc.shape[1]):
        tmp[f'logvar_z{idx}'] = np.square(np.exp(logvar_acc[..., idx]))

    df_logvar = pd.DataFrame(tmp)
    df_logvar['label'] = df_logvar['label'].astype('category')


    tmp = {}
    for idx in range(0, model.dec_fc1.weight.T.shape[0]):
        tmp[f'w{idx}'] = list(model.dec_fc1.weight.T[idx, ...].data.cpu().numpy())

    df_dec1_weights = pd.DataFrame(tmp)
    
    return df_mu, df_logvar, df_dec1_weights

def plot_data_boxplots(df_mu, df_logvar, df_dec1_weights, baseline_figsize=(1.2, 6)):
    figwidth, figheight = baseline_figsize
    df_mu2 = df_mu.melt(['label'])
    plt.figure(figsize=(int(figwidth * LATENT_SIZE), figheight))
    sns.boxplot(x='variable', y='value', data=df_mu2)
    plt.title("Distribution of $\mu$ in latent space")

    df_logvar2 = df_logvar.melt(['label'])
    plt.figure(figsize=(int(figwidth * LATENT_SIZE), figheight))
    sns.boxplot(x='variable', y='value', data=df_logvar2)
    plt.title("Distribution of $\sigma^2$ in latent space")

    df_dec1_weights2 = df_dec1_weights.melt()
    plt.figure(figsize=(int(figwidth * LATENT_SIZE), figheight))
    sns.boxplot(x='variable', y='value', data=df_dec1_weights2)
    plt.title("Weights going to decoder from latent space")

def walk_in_latent_space(latent_space_abs_limit=3, sqrt_sample_count=20, latent_size=2, dimensions_to_walk=(0, 1), figsize=(16, 16)):
    dim1, dim2 = dimensions_to_walk
    canvas = np.zeros((sqrt_sample_count * 28, sqrt_sample_count * 28))

    synthetic_representations = np.zeros((sqrt_sample_count * sqrt_sample_count, latent_size))

    synthetic_representations[..., dim1] = np.linspace(-latent_space_abs_limit, latent_space_abs_limit, num=sqrt_sample_count * sqrt_sample_count)
    synthetic_representations[..., dim2] = np.linspace(-latent_space_abs_limit, latent_space_abs_limit, num=sqrt_sample_count * sqrt_sample_count)

    recons = model.decode(torch.from_numpy(synthetic_representations).float().to('cuda'))

    for idx in range(0, sqrt_sample_count * sqrt_sample_count):
        x, y = np.unravel_index(idx, (sqrt_sample_count, sqrt_sample_count))
        canvas[y*28:((y+1) * 28), x*28:((x+1) * 28)] = recons[idx, ...].view(28, 28).data.cpu().numpy()

    plt.figure(figsize=figsize)
    plt.imshow(canvas)
```


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

Implement a DCGAN with the following architecture:
    
* Generator
    * Layer 1 - Number of output channels = 512, kernel size = 4, stride = 1
    * Layer 2 - Number of output channels = 256, kernel size = 4, stride = 2, padding = 1
    * Layer 3 - Number of output channels = 128, kernel size = 4, stride = 2, padding = 1
    * Layer 4 - Number of output channels = 64, kernel size = 4, stride = 2, padding = 1
    * Layer 5 - Number of output channels = 1, kernel size = 4, stride = 2, padding = 1

* Discriminator
    * Layer 1 - Broj izlaznih konvolucija = 64, kernel size = 4, stride = 2, padding = 1
    * Layer 2 - Broj izlaznih konvolucija = 128, kernel size = 4, stride = 2, padding = 1
    * Layer 3 - Broj izlaznih konvolucija = 256, kernel size = 4, stride = 2, padding = 1
    * Layer 4 - Broj izlaznih konvolucija = 512, kernel size = 4, stride = 2, padding = 1
    * Layer 5 - Broj izlaznih konvolucija = 1, kernel size = 4, stride = 1, padding = 0

Use kernel size [4,4] in all convolutions except for the output layer of the discriminator. The number of channels from the input to the output layers should be G: 512, 256, 128, 1 and D: 64, 128, Generator input $$\mathbf z$$ should have 100 elements obeying the normal distribution $$ N (0,1) $$. Use MNIST numbers scaled to size 32x32 as training set and train the network for at least 20 epochs. In each iteration, perform optimization of the generator and one optimization of the discriminator with one mini-batch each. Use a tanh activation function for the generator output and sigmoid activation for the discriminator output.

**Subtasks:**

 1. Visualize the results of generating 100 new samples from random variables $$ \ mathbf z $$. Compare the results with samples generated by VAE.
 2. In one iteration, use two mini-batches to train the generator and only one mini-batch to train the discriminator. Visualize generated samples. Repeat the same procedure with two mini-batches for the discriminator and one for the generator. Comment on the results.
 3. Turn off batch normalization in both networks. Comment on the results.

Use the following template:

**REMARK**: In addition to filling out the missing code, the template should be tailored as needed, and can be customized freely. So please **be especially careful with the claims that some of the code is not working for you!**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

from torchvision import datasets, transforms
import tqdm
from torchvision.utils import make_grid

import torch.distributions as tdist

import numpy as np
import tqdm

import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

def prepare_data_loaders(batch_size=32):
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files', train=True, download=True, 
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.Resize((64, 64)),
                                   torchvision.transforms.ToTensor()
                               ])), batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./files', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                   torchvision.transforms.Resize((64, 64)),
                                       torchvision.transforms.ToTensor()
                                   ])), batch_size=batch_size)
    
    return train_loader, test_loader

class Generator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()

        self.latent_size = latent_size
        
        ???


    def forward(self, x):
        ???

        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        ???

    def forward(self, x):
        ???

        return x

def weights_init(w):
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)

dmodel = Discriminator()
gmodel = Generator(100)

dmodel.apply(weights_init)
gmodel.apply(weights_init)

def train(gmodel: Generator, dmodel: Discriminator, n_epochs=10, log_epochs=1, batch_size=32, learning_rate=1e-3, device='cpu'):
    train_loader, test_loader = prepare_data_loaders(batch_size=batch_size)
    
    gmodel = gmodel.to(device)
    dmodel = dmodel.to(device)
    
    gmodel.train()
    dmodel.train()
    
    criterion = nn.BCELoss()
    
    g_optim = optim.Adam(gmodel.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    d_optim = optim.Adam(dmodel.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    for epoch_idx in range(0, n_epochs):
        
        g_loss, d_loss = 0, 0
        
        for image_data, _ in tqdm.tqdm(train_loader):
            # discriminator update
            dmodel.zero_grad()
            
            # real data pass
            image_data = image_data.to(device)
            
            batch_size = image_data.shape[0]
            labels = torch.ones(batch_size, device=device).float()
            
            d_output = dmodel(image_data)
            d_err_real = criterion(d_output, labels)
            d_err_real.backward()
            d_loss += d_err_real.item() / batch_size
            
            # fake data pass
            noise = torch.randn(batch_size, gmodel.latent_size, 1, 1, device=device)
            fake_image_data = gmodel(noise)
            labels = torch.zeros(batch_size, device=device).float()
            
            d_output = dmodel(fake_image_data.detach())
            d_error_fake = criterion(d_output, labels)
            d_error_fake.backward()
            d_loss += d_error_fake.item() / batch_size
    
            d_optim.step()
            
            # generator update
            gmodel.zero_grad()
            
            labels = torch.ones(batch_size, device=device)
            d_output = dmodel(fake_image_data)
            g_error = criterion(d_output, labels)
            g_error.backward()
            g_loss += g_error.item() / batch_size 
            g_optim.step()
            
        if (epoch_idx + 1) % log_epochs == 0:
            print(f"[{epoch_idx+1}/{n_epochs}]: d_loss = {d_loss:.2f} g_loss {g_loss:.2f}")

    gmodel.eval()
    dmodel.eval()
    
    return gmodel, dmodel

gmodel, dmodel = train(gmodel, dmodel, n_epochs=15, batch_size=256, device='cuda')

random_sample = gmodel(torch.randn(100, 100, 1, 1).to('cuda')).view(100, 64, 64).data.cpu().numpy()

plt.figure(figsize=(16, 16))
for idx in range(0, 100):
    plt.subplot(10, 10, idx+1)
    plt.imshow(random_sample[idx, ...])
    plt.clim(0, 1)
    plt.axis('off')
```

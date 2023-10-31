---
layout: page_en
mathjax: true
permalink: /lab2en/
---

## Exercise 2: convolutional models 

- [Introduction](#intro)
- [Exercise overview](#vjezba)
- [Task 1: backprop through FC, ReLU and log softmax](#1zad)
- [Task 2: regularization](#2zad)
- [Task 3: PyTorch MNIST](#3zad)
- [Task 4: Pytorch CIFAR-10](#4zad)
- [Bonus tasks](#add)


<a name='intro'></a>
## Exercise 2: Convolutional models (CNN)

This exercise considers convolutional models.
Such models are appropriate for structured
data arranged in a special topology
which results in invariance to translation.
Images are typical exemplars of such data
since an object which defines the classification outcome
can appear at any image location.
If we feed a vectorized image to the input
of a fully connected layer,
each latent activation would see all image pixels.
Such model would allow latent activations
to specialize for particular image locations
and therefore hinder generalization
and promote overfitting.
Such organization would also require
an extraordinary large number of parameters
due to large number of pixels.
For example, a single activation
sensing a 200x200 colour image
would need to have 3\*200\*200=120,000 weight parameters.
It is clear that under such circumstances
we would run out of memory very soon.

We see that it would be much better
if each activation sensed
only a small portion of the input image.
That would favour specialization
to local neighbourhoods and
substantially decrease the number of parameters.
Multiple activations could operate
on different local neighbourhoods
with the same set of parameters.
Thus, translated objects would cause
translated latent activations
(this property is called translation equivariance).

The desired properties can be achieved 
by convolving the input layer
with a trainable convolution kernel (or filter).
In this setup, activations in the first latent layer
encode local features such as edges and or corners.
Later layers hierarchically build
more sophisticated representations
which respond to object parts and complete objects.
These activations require a larger receptive field
which means they have to be sensitive
to a larger neighbourhood of the original image.
This can be achieved without increasing
the number of model parameters,
by introducing pooling layers which
reduce the spatial extent of the data representation.
Contemporary architectures typically
terminate the feature extraction phase
with a global pooling operation 
in order to achieve translation invariance.

To conclude, convolutional models leverage
the following three important ideas:
sparse connectivity, parameter sharing, 
and translation equivariance of latent representations.
These ideas decrease number of parameters
without sacrificing expressiveness of the model, 
discourage overfitting and promote generalization.

<div class="fig figcenter fighighlight">
  <img src="/assets/lab2/convnet1.png" width="100%">
  <!--<img src="/assets/lab2/convnet2.png" width="30%">-->
  <div class="figcaption figcenter">A typical
    convolutional model for image classification
    contains a succession of convolutions and poolings.
    At the end we arrive to global image representation
    which is mapped to class posteriors
    with a fully connected layer.
  </div>
</div>

<a name='vjezba'></a>

## Exercise overview

The required dependencies are:
PyTorch, NumPy, [Cython](http://cython.org),
[matplotlib](http://matplotlib.org/) and
[scikit-image](http://scikit-image.org/).
Be careful to pick versions for Python 3.

For the purpose of tasks 1 and 2
we have designed a minimal deep learning framework.
Your task is to complete the missing pieces
and train a convolutional model on the MNIST dataset.
The framework can be downloaded 
[here](https://github.com/ivankreso/fer-deep-learning/tree/master/lab2).
The file `layers.py` includes definitions of layers
which are found in a typical CNN.
Each layer contains one forward pass method
and two backprop methods.
The `forward` method should perform a forward pass
through the layer and return the result.
The methods `backward_inputs` and `backward_params`
perform the backprop over layer inputs and layer parameters.
The method `backward_inputs` calculates 
partial derivatives with respect to layer inputs
(\\( \frac{∂L}{∂\mathbf{x}} \\) where
\\(\mathbf{x}\\) correspond to layer inputs).
The method  `backward_params` calculates 
partial derivatives with respect to
layer parameters (\\( \frac{∂L}{∂\mathbf{w}} \\)
where vector \\(\mathbf{w}\\) corresponds to
all layer parameters).

<a name='1zad'></a>

### Task 1: backprop through FC, ReLU and log softmax

Complete implementations of the fully connected layer,
hinge nonlinearity layer and the cross entropy loss layer
(`FC`, `ReLU`, `SoftmaxCrossEntropyWithLogits`).
The cross entropy loss layer calculates divergence
between the exact distribution and the distribution
predicted by the model:

$$
L = - \sum_{i=1}^{C} y_i
        \log(\mathrm{softmax}_i(\mathbf{x})). \\
$$

C stands for the number of classes,
\\( \mathbf{x} \\) is the input to the softmax function
(which may be called classification score or logit),
\\( \mathbf{y} \\) is the groundtruth distribution
of the example over all classes
(often given as a one-hot vector), 
while \\( \mathrm{softmax}_i(\mathbf{x}) \\)
is the softmax output for the class \\(i\\).
For simplicity, we have shown the loss
for only one example while in practice
it will often be average loss over the whole batch.
To perform backprop, we first need to calculate
the gradient of this function with respect to input
\\( \frac{∂L}{∂\mathbf{x}} \\).
The derivation of this gradient may be simplified
by plugging-in the definition of the softmax:

$$
\log(\mathrm{softmax}_i(x)) =
  \log \left(\frac{e^{x_i}}{\sum_{j=1}^{C} e^{x_j}}\right) =
    x_i - log \sum_{j=1}^{C} e^{x_j} \\
L =
  - \sum_{i=1}^{C} y_i
    \left(x_i - log \sum_{j=1}^{C} e^{x_j}\right) =
  - \sum_{i=1}^{C} y_i x_i + \log
    \left(\sum_{j=1}^{C} e^{x_j}\right)
    \sum_{i=1}^{C} y_i \;\; ; \;\;\;\;
    \sum_{i=1}^{C} y_i = 1 \\
L = \log \left(\sum_{j=1}^{C} e^{x_j}\right)
  - \sum_{i=1}^{C} y_i x_i \\
$$

<!---
\sum_{i=1}^{C} y_i log(s_j(x)) \\
L = log \left(\sum_{j=1}^{C} e^{x_j}\right) - \sum_{i=1}^{C} y_i x_i \\
-->

Now we can determine the derivative
of the goal function with respect
to single classification score
( x_k \\):

$$
\frac{∂L}{∂x_k} =
  \frac{∂}{∂x_k} \log \left(\sum_{j=1}^{C} e^{x_j}\right)
    - \frac{∂}{∂x_k} \sum_{i=1}^{C} y_i x_i \\
\frac{∂}{∂x_k} log \left(\sum_{j=1}^{C} e^{x_j}\right)
  = \frac{1}{\sum_{j=1}^{C} e^{x_j}} \cdot e^{x_k}
  = s_k(\mathbf{x}) \\
\frac{∂L}{∂x_k} = s_k(\mathbf{x}) - y_k \\
$$

Finally, we obtain the gradient 
with respect to layer inputs
as the difference between
the model distribution 
and the exact distribution:

$$
\frac{∂L}{∂\mathbf{x}} = s(\mathbf{x}) - \mathbf{y} \\
$$

In order to setup your environment, 
set the appropriate paths in variables
`DATA_DIR` and `SAVE_DIR` and compile
the cython module `im2col_cython.pyx` 
with `python3 setup_cython.py build_ext --inplace`.
Go over the impementations of functions `col2im_cython`
and `im2col_cython` and find out how these
functions are used.

Test implementations of your layers
by invoking the script `check_grads.py`.
A satisfactory error should be less than \\(10^{-5}\\)
if you use double precision.
Examine the source code of that script
because it will be very useful for the third exercise.
Consider why, when training deep models, 
we prefer to use analytical gradients rather than numerical gradients.


Examine and outline the model
defined by the object `net` in the script `train.py`.
Determine the tensor sizes and
the number of parameters in each layer. 
Calculate the size of the feature receptive
field in the last (second) convolutional layer.
Estimate the total amount of memory
required to store activations needed 
for backpropagation when training 
with mini-batches of 50 images.

Finally, start the training of a convolutional model
by invoking the script `train.py`.
The script will download the MNIST dataset
to the `SAVE_DIR` directory.
During training you can monitor the visualization
of the learnt filters which are saved
in the `SAVE_DIR` directory.
Since each weight correspond to a single image pixel,
we recommend to turn off antialiasing
for best viewing experience.
We recommend to use geeqie on Linux.

<a name='2zad'></a>

### Task 2: regularization

In this task you need to add support for 
L2 parameter regularization.
Complete the implementation of the `L2Regularizer` layer
and train the regularized model by invoking `train_l2reg.py`.
Examine effects of the regularization hyper-parameter
by training three different models with
\\( \lambda = 1e^{-3}, \lambda=1e^{-2}, \lambda=1e^{-1} \\)
and comparing the filters from the first layer
and the achieved accuracy.

<!---
Minimalno je potrebno izmijeniti update\_params metode.
Dodajte novi sloj L2Loss koji na ulazu forward metode prima tenzor težine te
implementirajte odgovarajući unaprijedni i unazadni prolazak.
Dodajte novi sloj MultiLoss koji na ulaz prima listu funkcija cilja te računa njihov zbroj.
Konačna funkcija cilja je sada MultiLoss koji u našem slučaju na ulazu treba dobiti
listu u kojoj se nalazi funkcija cilja unakrsne entropije i L2 regularizacije težina
konvolucijskih i potpuno povezanih slojeva.
-->

<div class="fig figcenter fighighlight">
  <img src="/assets/lab2/filters1_big.png" width="80%">
  <img src="/assets/lab2/filters2_big.png" width="80%">
  <div class="figcaption">Random initializations
    of the filters from the first layer  (top)
    and the trained filters (bottom)
    with lambda = 0.01.</div>
</div>

<a name='3zad'></a>

### Task 3 - PyTorch MNIST
In PyTorch, define and train a model equivalent
to the regularized model specified in the 2nd task.
Use the same architecture and learning parameters
to reproduce the results.
Implement the convolution using [`torch.nn.Conv2d`](https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d)
or [`torch.nn.functional.conv2d`](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.conv2d).
Below, we provide an example of using the `torch.nn.Conv2d` class for convolution.

```python
import torch
from torch import nn
 
class CovolutionalModel(nn.Module):
  def __init__(self, in_channels, conv1_width, ..., fc1_width, class_count):
    self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding=2, bias=True)
    # ostatak konvolucijskih slojeva i slojeva sažimanja
    ...
    # potpuno povezani slojevi
    self.fc1 = nn.Linear(..., fc1_width, bias=True)
    self.fc_logits = nn.Linear(fc1_width, class_count, bias=True)

    # parametri su već inicijalizirani pozivima Conv2d i Linear
    # ali možemo ih drugačije inicijalizirati
    self.reset_parameters()

  def reset_parameters(self)
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear) and m is not self.fc_logits:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    self.fc_logits.reset_parameters()

  def forward(self, x):
    h = self.conv1(x)
    ...
    h = torch.relu(h)  # može i h.relu() ili nn.functional.relu(h)
    ...
    h = h.view(h.shape[0], -1)
    h = self.fc1(h)
    h = torch.relu(h)
    logits = self.fc_logits(h)
    return logits

```

If you want to use `torch.nn.functional.conv2d`,
pay attention to manually defining the parameters of 
type [`torch.nn.Parameter`](https://pytorch.org/docs/stable/nn.html#torch.nn.Parameter).
You can refer to the following [example](https://pytorch.org/docs/stable/notes/extending.html#adding-a-module).
Ensure the appropriate tensor dimensionality for the convolution kernels and bias tensors.

During the training, visualize the filters in the first layer
as in the previous exercise. 
After each epoch, store the filters and loss in a file
(or use [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html)).

At the end of the training, show the loss evolution across epochs using Matplotlib.

Unlike Exercise 1, you can use `torch.nn.data.DataLoader`
 for iterating and sampling mini-batches.
You can find more information [here](https://pytorch.org/docs/stable/data.html) .

<a name='4zad'></a>

### Task 4: PyTorch CIFAR 10

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
dataset contains 50000 images for training and validation,
and 10000 test images.
The images have dimensions 32x32
and they belong to 10 classes.
Download the dataset version for Python
[here](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
or by using [`torchvision.datasets.CIFAR10`](https://pytorch.org/docs/stable/torchvision/datasets.html#cifar).
Use the following code to load and prepare the dataset.

```python
import os
import pickle
import numpy as np

def shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y

def unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='latin1')
  fo.close()
  return dict

DATA_DIR = '/path/to/data/'

img_height = 32
img_width = 32
num_channels = 3
num_classes = 10

train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
train_y = []
for i in range(1, 6):
  subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
  train_x = np.vstack((train_x, subset['data']))
  train_y += subset['labels']
train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
train_y = np.array(train_y, dtype=np.int32)

subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
test_y = np.array(subset['labels'], dtype=np.int32)

valid_size = 5000
train_x, train_y = shuffle_data(train_x, train_y)
valid_x = train_x[:valid_size, ...]
valid_y = train_y[:valid_size, ...]
train_x = train_x[valid_size:, ...]
train_y = train_y[valid_size:, ...]
data_mean = train_x.mean((0, 1, 2))
data_std = train_x.std((0, 1, 2))

train_x = (train_x - data_mean) / data_std
valid_x = (valid_x - data_mean) / data_std
test_x = (test_x - data_mean) / data_std

train_x = train_x.transpose(0, 3, 1, 2)
valid_x = valid_x.transpose(0, 3, 1, 2)
test_x = test_x.transpose(0, 3, 1, 2)
```


Your task is to train a convolutional model in PyTorch.
We propose a simple model which should yield
about 70\% accuracy in image classification.

```
conv(16,5) -> relu() -> pool(3,2) -> conv(32,5) -> relu() -> pool(3,2) -> fc(256) -> relu() -> fc(128) -> relu() -> fc(10)
```
Here `conv(16,5)` represents a convolution
with 16 feature maps and filter dimensions 5x5,
`pool(3,2)` is a max-pooling layer operating
on patches 3x3 and the stride 2.


Write the function `evaluate` which
compares the predicted and correct class indices
and determines the following
classification performance indicators:
overall classification accuracy,
confusion matrix, as well as
precision and recall for particular classes.
In the implementation, first determine the confusion matrix,
and then use it to determine all other indicators.
During training, invoke `evaluate`
after each epoch both on the training 
and on the validation dataset,
and graph the average loss, the training rate
and overall classification accuracy.
We recommend that function receives
the data, the correct class indices,
and the required pytorch operations.
Be careful not to invoke the training operation.
The function should output
the recovered indicators to the console.

<div class="fig figcenter fighighlight">
  <img src="/assets/lab2/training_plot.png" width="100%">
  <div class="figcaption figcenter">An example of the
training graph for the given model with a batch size of 50.
    when the training proceeds well.</div>
</div>

Visualize random initializations
and the trained filters from the first layer, 
which you can access through `conv.weight`.
We supply an example of how that might look like below.

```python
net = ConvNet()

draw_conv_filters(0, 0, net.conv1.weight.detach().numpy(), SAVE_DIR)
```

We also provide code which you can use
for visualization:

```python
def draw_conv_filters(epoch, step, weights, save_dir):
  w = weights.copy()
  num_filters = w.shape[0]
  num_channels = w.shape[1]
  k = w.shape[2]
  assert w.shape[3] == w.shape[2]
  w = w.transpose(2, 3, 1, 0)
  w -= w.min()
  w /= w.max()
  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border
  img = np.zeros([height, width, num_channels])
  for i in range(num_filters):
    r = int(i / cols) * (k + border)
    c = int(i % cols) * (k + border)
    img[r:r+k,c:c+k,:] = w[:,:,:,i]
  filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
  ski.io.imsave(os.path.join(save_dir, filename), img)
```

<div class="fig figcenter fighighlight">
  <img src="/assets/lab2/cifar_filters1.png" width="80%">
  <img src="/assets/lab2/cifar_filters2.png" width="80%">
  <div class="figcaption figcenter">CIFAR-10: 
    random initializations (top) and the learned filters
    in the first layer (bottom) with
    regularization lambda = 0.0001.</div>
</div>

Visualize 20 incorrectly classified images
with the largest loss and output the correct class
and the top 3 predicted classes.
Pay attention that in order to visualize image,
you first need to undo the normalization
of the mean value and variance:

```python
import skimage as ski
import skimage.io

def draw_image(img, mean, std):
  img = img.transpose(1, 2, 0)
  img *= std
  img += mean
  img = img.astype(np.uint8)
  ski.io.imshow(img)
  ski.io.show()
```
We provide the code for producing graphs below:

```python
def plot_training_progress(save_dir, data):
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

  linewidth = 2
  legend_size = 10
  train_color = 'm'
  val_color = 'c'

  num_points = len(data['train_loss'])
  x_data = np.linspace(1, num_points, num_points)
  ax1.set_title('Cross-entropy loss')
  ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax1.legend(loc='upper right', fontsize=legend_size)
  ax2.set_title('Average class accuracy')
  ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax2.legend(loc='upper left', fontsize=legend_size)
  ax3.set_title('Learning rate')
  ax3.plot(x_data, data['lr'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='learning_rate')
  ax3.legend(loc='upper left', fontsize=legend_size)

  save_path = os.path.join(save_dir, 'training_plot.png')
  print('Plotting in: ', save_path)
  plt.savefig(save_path)
```

```python
plot_data = {}
plot_data['train_loss'] = []
plot_data['valid_loss'] = []
plot_data['train_acc'] = []
plot_data['valid_acc'] = []
plot_data['lr'] = []

for epoch in range(num_epochs):
    X, Yoh = shuffle_data(train_x, train_labels)
    X = torch.FloatTensor(X)
    Yoh = torch.FloatTensor(Yoh)
    for batch in range(n_batch):
        # broj primjera djeljiv s veličinom grupe bsz
        batch_X = X[batch*bsz:(batch+1)*bsz, :]
        batch_Yoh = Yoh[batch*bsz:(batch+1)*bsz, :]

        loss = model.get_loss(batch_X, batch_Yoh)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch%100 == 0:
            print("epoch: {}, step: {}/{}, batch_loss: {}".format(epoch, batch, n_batch, loss))

        if batch%200 == 0:
            draw_conv_filters(epoch, batch, model.conv1.weight.detach().cpu().numpy(), SAVE_DIR)


    train_loss, train_acc = evaluate(model, train_x, train_labels)
    val_loss, val_acc = evaluate(model, valid_x, valid_labels)

    plot_data['train_loss'] += [train_loss]
    plot_data['valid_loss'] += [val_loss]
    plot_data['train_acc'] += [train_acc]
    plot_data['valid_acc'] += [val_acc]
    plot_data['lr'] += [lr_scheduler.get_lr()]
    lr_scheduler.step()

plot_training_progress(SAVE_DIR, plot_data)
```


If you have access to a GPU, you might want 
to try obtaining better results with a
more powerful model.
In that case, 
[here](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130)
you can find a review of state-of-the-art
results on this dataset.
As you see, best approaches achieve around 96%
overall classification accuracy.
Two important tricks to achieve this are
image upsampling and jittering.
Image upsampling ensures that
early convolutions detect
very low level features,
while jittering prevents overfitting.
Without these techniques, it will be very hard for you
to achieve more than 90%
overall classification accuracy.


### Bonus task - Multiclass hinge loss
This task considers training a model for CIFAR images
with an alternative loss formulation
that was not covered in lectures.
The goal is to replace the cross-entropy loss
with a multi-class version of the hinge loss.
An explanation of this loss can be found 
[here](http://cs231n.github.io/linear-classify/#svm).
The task requires solving for all points
by using basic PyTorch tensor operations
and comparing the achieved results.

Hint: The interface of the new loss function might look like this:

```
def multiclass_hinge_loss(logits: torch.Tensor, target: torch.Tensor, delta=1.):  
    """
        Args:
            logits: torch.Tensor with shape (B, C), where B is batch size, and C is number of classes.
            target: torch.LongTensor with shape (B, ) representing ground truth labels.
            delta: Hyperparameter.
        Returns:
            Loss as scalar torch.Tensor.
    """
```

You can start by separating the output
of the last fully connected layer into
a vector of logits for the correct classes
and a matrix of logits for incorrect classes.
You can accomplish this by using the function
[torch.masked_select](https://pytorch.org/docs/stable/torch.html#torch.masked_select),
with the mask specified as the regular or
inverted version of the matrix with one-hot encoded labels.
Now, the difference between the matrix of incorrect class
logits and the vector of correct class logits
can be computed using simple subtraction
because PyTorch automatically broadcasts the lower-rank operand.
Ensure to reshape all tensors into the correct shapes
as the function `torch.masked_select`
 returns a one-dimensional tensor.
You can compute the element-wise maximum
using an appropriate variant of the function
[torch.max](https://pytorch.org/docs/stable/torch.html#torch.max).

<a name='add'></a>

### Additional materials

- [Deep learning book](http://www.deeplearningbook.org/contents/convnets.html)
- [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io)

---
layout: page
mathjax: true
permalink: /lab2/
---

- [Konvolucijska neuronska mreža](#cnn)
- [Vježba](#vjezba)
  - [1. zadatak](#1zad)
  - [2. zadatak](#2zad)
  - [3. zadatak](#3zad)
  - [4. zadatak](#4zad)
- [Dodatni materijali](#add)


<a name='cnn'></a>

## Konvolucijska neuronska mreža (CNN)

U ovoj vježbi bavimo se konvolucijskim neuronskim mrežama. Konvolucijske mreže zamišljene
su za obradu podataka koji imaju posebnu topologiju gdje je osobito važno ostvariti
invarijantnost na translaciju (problem miješanja dimenzija u podacima).
Dobar primjer takvih podataka su slike gdje se obično isti objekt može pojaviti na bilo kojem
mjestu unutar slike. Ako bismo takvu sliku poslali na ulaz potpuno povezanog sloja tada bi
jedan neuron vidio sve piksele slike. Takav pristup bi omogućio svim neuronima da se
specijaliziraju za značajke objekta u djelovima slike u kojem se objekt pojavio što bi
na kraju rezultiralo prenaučenosti i model bi loše generalizirao.
Osim toga dodatni problem je što slike obično sadrže puno piksela. Na primjer, prosječne dimenzije
slike iz poznatog dataseta ImageNet iznose 3x200x200 što znači da bi u tom slučaju jedan
neuron u prvom sloju morao imati 3\*200\*200=120,000 težina. S većim brojem neurona bismo
jako brzo ostali bez memorije.

Vidimo da bismo bili u puno boljoj situaciji ako bismo ostvarili da svaki neuron
djeluje lokalno na samo jedan dio slike. Na taj način bi neuron imao rijetku povezanost
sa slikom što bi uvelike smanjilo broj težina. Ideja je da neuroni imaju jako
mala receptivna polja što znači da bi u prvom sloju mogli opisivati samo značajke jako
niske razine poput linija i rubova. Kasniji slojevi bi imali sve veće receptivno polje
što bi omogućilo da hijerarhijski grade kompleksnije značajke na temelju jednostavnijih.
Dodatno, budući da želimo postići invarijantnost na translacije unutar slike
i dalje želimo da svaki neuron djeluje nad čitavom slikom. To možemo ostvariti tako da
za svaki neuron umjesto jednog izlaza kao do sada imamo više izlaza. Svaki izlaz tada
bi odgovarao odzivu na drugom položaju u slici. Ovime smo postigli da se parametri jednog neurona
dijele preko čitave slike.
Neurone u konvolucijskim slojevima mreže obično nazivamo i filtrima.
Konvolucijske mreže koriste tri važne ideje: rijetku povezanost, dijeljenje parametara i
ekvivarijantnost reprezentacije.


**Konvolucija - demonstracija**. Ispod se nalazi demonstracija konvolucijskog sloja.
Plavo - ulazni tenzor, crveno - tensor filtara, zeleno - izlazni tenzor.
Svi tensozi izrezani su po dubini radi lakše vizualizacije.
Preuzeto s [CS231n](http://cs231n.github.io/convolutional-networks/).
<!---
The input volume is of size \\(W_1 = 5, H_1 = 5, D_1 = 3\\), and the CONV layer parameters are \\(K = 2, F = 3, S = 2, P = 1\\).
That is, we have two filters of size \\(3 \times 3\\), and they are applied with a stride of 2.
Therefore, the output volume size has spatial size (5 - 3 + 2)/2 + 1 = 3.
Moreover, notice that a padding of \\(P = 1\\) is applied to the input volume, making the outer border of the input volume zero.
The visualization below iterates over the output activations (green), and shows that each element is computed by elementwise multiplying the highlighted input (blue) with the filter (red), summing it up, and then offsetting the result by the bias.
-->

<div class="fig figcenter fighighlight">
  <iframe src="/assets/conv-demo/index.html" width="100%" height="700px;" style="border:none;"></iframe>
  <div class="figcaption"></div>
</div>


<a name='vjezba'></a>

## Vježba

Kod za prva dva zadatka nalazi se [ovdje](https://github.com/ivankreso/fer-deep-learning/tree/master/lab2).
U datoteci `layers.py` nalaze se slojevi od kojih se tipično sastoji CNN.
Svaki sloj sadrži dvije metode potrebne za izvođenje backpropagation algoritma.
Metoda `forward` izvodi unaprijedni prolazak kroz sloj i vraća rezultat.
Metode `backward_inputs` i `backward_params` izvode unazadni prolazak.
Metoda `backward_inputs` računa gradijent s obzirom na ulazne podatke (\\( \frac{∂L}{∂\mathbf{x}} \\) gdje je \\(\mathbf{x}\\) ulaz u sloj).
Metoda  `backward_params` računa gradijent s obzirom na parametre sloja (\\( \frac{∂L}{∂\mathbf{w}} \\) gdje vektor \\(\mathbf{w}\\) vektor predstavlja sve parametre sloja)).

<a name='1zad'></a>

### 1. zadatak
Dovršite implementacije potpuno povezanog sloja, sloja nelinearnosti 
te funkcije gubitka u razredima `FC`, `ReLU` i `SoftmaxCrossEntropyWithLogits`.
Podsjetimo se, funkcija cilja unakrsne entropije računa udaljenost između
točne distribucije i distribucije koju predviđa model i definirana je kao:

$$
L = - \sum_{i=1}^{C} y_i log(s_j(\mathbf{x})) \\
$$

gdje je C broj razreda, \\( \mathbf{x} \\) ulazni primjer u vektorkom obliku,
\\( \mathbf{y} \\) točna distribucija preko svih razreda za dani primjer (najčešće one-hot vektor), a \\( s_j(\mathbf{x}) \\)
izlaz Softmax funkcije za razred \\(j\\).
Da biste izveli unazadni prolazak kroz sloj potrebno je najprije izračunati
gradijent ove funkcije s obzirom na ulaz \frac{∂L}{∂\mathbf{x}}.
Postupak derivacije možemo pojednostavniti tako da uvrstimo definiciju Softmax funkcije:

$$
log(s_i(x)) = log \left(\frac{e^{x_i}}{\sum_{j=1}^{C} e^{x_j}}\right) = x_i - log \sum_{j=1}^{C} e^{x_j} \\
L = - \sum_{i=1}^{C} y_i \left(x_i - log \sum_{j=1}^{C} e^{x_j}\right) = - \sum_{i=1}^{C} y_i x_i + log \left(\sum_{j=1}^{C} e^{x_j}\right) \sum_{i=1}^{C} y_i \;\; ; \;\;\;\; \sum_{i=1}^{C} y_i = 1 \\
L = log \left(\sum_{j=1}^{C} e^{x_j}\right) - \sum_{i=1}^{C} y_i x_i \\
$$

<!---
\sum_{i=1}^{C} y_i log(s_j(x)) \\
L = log \left(\sum_{j=1}^{C} e^{x_j}\right) - \sum_{i=1}^{C} y_i x_i \\
-->


Sada možemo jednostavno izračunati derivaciju funkcije cilja s obzirom na ulazni skalar \\( x_k \\):

$$
\frac{∂L}{∂x_k} = \frac{∂L}{∂x_k} log \left(\sum_{j=1}^{C} e^{x_j}\right) - \frac{∂L}{∂x_k} \sum_{i=1}^{C} y_i x_i \\
\frac{∂L}{∂x_k} log \left(\sum_{j=1}^{C} e^{x_j}\right) = \frac{1}{\sum_{j=1}^{C} e^{x_j}} \cdot e^{x_k} = s_k(\mathbf{x}) \\
\frac{∂L}{∂x_k} = s_k(\mathbf{x}) - y_k \\
$$

Konačno, gradijent s obzirom na sve ulazne podatke dobijemo tako da izračunamo razliku između vektora distribucije iz modela i
točne distribucije:

$$
\frac{∂L}{∂\mathbf{x}} = s(\mathbf{x}) - \mathbf{y} \\
$$

Kako biste bili sigurni da ste ispravno napisali sve slojeva testirajte gradijente pozivom skripte `check_grads.py`.
Zadovoljavajuća relativna greška bi trebala biti manja od \\(10^{-5}\\) ako vaši tenzori imaju dvostruku preciznost.
Napokon, pokrenite učenje modela pozivom skripte `train.py`. Napomena: najprije postavite odgovarajuće puteve u varijable
`DATA_DIR` i `SAVE_DIR`;

Tijekom učenja možete promatrati vizualizaciju filtara koji se spremaju u `SAVE_DIR` direktorij.
Budući da svaka težina odgovara jednom pikselu slike u vašem pregledniku isključite automatsko glađenje slike da biste mogli bolje vidjeti.
Preporuka je da na Linuxu koristite preglednik Geeqie.

<a name='2zad'></a>

### 2. zadatak
U ovom zadatku trebate dodati podršku za L2 regularizaciju parametara.
Dovršite implementaciju `L2Regularizer` sloja te naučite regularizirani model iz
prethodnog zadatka koji se nalazi u `train_l2reg.py`.
Igrajte se s regularizacijskim parametrom tako da naučite
tri različite mreže \\( \lambda = 0, lambda=1e^{-3}, lambda=1e{-1} \\)
te usporedite naučene filtre u prvom sloju i dobivenu točnost.

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
  <img src="/assets/lab2/filters1_big.png" width="90%">
  <img src="/assets/lab2/filters2_big.png" width="90%">
  <div class="figcaption">Slučajno inicijalizirani filtri u prvom sloju na početku učenja (iznad) i naučeni filtri (ispod).</div>
</div>

<a name='3zad'></a>

### 3. zadatak - usporedba s Tensorflowom
U Tensorflowu definirajte i naučite model koji je ekvivalentan modelu iz 2. zadatka.
Korisite identičnu arhitekturu i parametre učenja.
Kako biste u graf dodali operaciju konvolucije koristite `tf.nn.conv2d` ili `tf.contrib.layers.convolution2d`.
Prije toga proučite službenu dokumentaciju vezanu za [konvoluciju](https://www.tensorflow.org/versions/master/api_docs/python/nn.html#convolution).

<!---
Dodajte u model normalizaciju podataka po slojevima nakon svakog konvolucijskog sloja ([Batch
normalization](https://arxiv.org/abs/1502.03167)). To najlakše možete
napraviti tako da konvoluciji zadate `tf.contrib.layers.batch_norm`
kao parametar normalizacije kako je prikazano ispod:
-->

```
import tensorflow.contrib.layers as layers

...

def build_model(inputs, labels, num_classes, is_training):
  weight_decay = 1e-3
  bn_params = {
      # Decay for the moving averages.
      'decay': 0.999,
      'center': True,
      'scale': True,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
      # None to force the updates during train_op
      'updates_collections': None,
      'is_training': is_training
  }

  with tf.contrib.framework.arg_scope([layers.convolution2d],
      kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu,
      normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
      weights_initializer=layers.variance_scaling_initializer(),
      weights_regularizer=layers.l2_regularizer(weight_decay)):

    net = layers.convolution2d(inputs, conv1sz, scope='conv1')
    ...
```

Usporedite nove i stare rezulate. Batch normalization bi trebao
poboljšati propagaciju gradijenata te ubrzati konvergenciju.


<a name='4zad'></a>

### 4. zadatak - Klasifikacija na CIFAR-10 skupu
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset sastoji se od 50000 slika za učenje i validaciju te 10000 slika za
testiranje dimenzija 32x32 podijeljenih u 10 razreda.
Najprije skinite dataset pripremljen za Python [ovdje](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz).
Iskorisite sljedeći kod kako biste učitali podatke i pripremili ih.

```
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

train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
train_y = []
for i in range(1, 6):
  subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
  train_x = np.vstack((train_x, subset['data']))
  train_y += subset['labels']
train_x = train_x.reshape((-1, img_height, img_width, num_channels))
train_y = np.array(train_y, dtype=np.int32)

subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
test_x = subset['data'].reshape((-1, img_height, img_width, num_channels)).astype(np.float32)
test_y = np.array(subset['labels'], dtype=np.int32)

valid_size = 5000
train_x, train_y = shuffle_data(train_x, train_y)
valid_x = train_x[:valid_size, ...]
valid_y = train_y[:valid_size, ...]
train_x = train_x[valid_size:, ...]
train_y = train_y[valid_size:, ...]
data_mean = train_x.mean((0,1,2))
data_std = train_x.std((0,1,2))

train_x = (train_x - data_mean) / data_std
valid_x = (valid_x - data_mean) / data_std
test_x = (test_x - data_mean) / data_std
```

Vaš zadatak je da u Tensorflowu naučite CNN i reproducirate naše rezulate na ovom skupu.
Arhitektura mreže treba biti sljedeća: TODO

<div class="fig figcenter fighighlight">
  <img src="/assets/lab2/training_plot.png" width="100%">
  <div class="figcaption">.</div>
</div>

Napišite funkciju `evaluate(y,yt)` koja na temelju predviđenih i točnih indeksa razreda određuje pokazatelje klasifikacijske performanse:
ukupnu točnost klasifikacije, matricu zabune (engl. confusion matrix) u kojoj retci odgovaraju točnim razredima a stupci predikcijama te mjere preciznosti
i odziva pojedinih razreda. U implementaciji prvo izračunajte matricu zabune, a onda sve ostale pokazatelje na temelju nje.

Tijekom učenja pozivajte funkciju `evaluate` nakon svake epohe na skupu za učenje i
validacijskom skupu te na grafu pratite sljedeće vrijednosti: prosječnu vrijednost
funkcije gubitka, stopu učenja, prosječnu klasifikacijsku preciznost te prosječni odziv.

Vizualizirajte naučene filtre u prvom sloju. Možete se poslužiti kodom za vizualizaciju
iz prve vježbe no morate ga izmijeniti tako da 3 kanala u prvom sloju kodirate kao RGB
boje slike.

Prikažite 20 netočno klasificiranih slika s najvećim gubitkom i ispod slike ispišite njihov točan razred
i top-3 razreda za koje je mreža dala najveću vjerojatnost.


Ispod se nalazi kod koji možete iskoristiti za crtanje grafova.

```
```

```
plot_data = {}
plot_data['train_loss'] = []
plot_data['valid_loss'] = []
plot_data['train_acc'] = []
plot_data['valid_acc'] = []
plot_data['lr'] = []
for epoch_num in range(1, num_epochs + 1):
  train_x, train_y = shuffle_data(train_x, train_y)
  for step in range(num_batches):
    offset = step * batch_size 
    batch_x = train_x[offset:(offset + batch_size), ...]
    batch_y = train_y[offset:(offset + batch_size)]
    feed_dict = {node_x: batch_x, node_y: batch_y}
    start_time = time.time()
    run_ops = [train_op, loss, logits, global_step, lr]
    ret_val = sess.run(run_ops, feed_dict=feed_dict)
    _, loss_val, logits_val, _, lr_val = ret_val
    duration = time.time() - start_time
    if (step+1) % 50 == 0:
      sec_per_batch = float(duration)
      format_str = 'epoch %d, step %d / %d, loss = %.2f (%.3f sec/batch)'
      print(format_str % (epoch_num, step+1, num_batches, loss_val, sec_per_batch))

  print('Train error:')
  train_loss, train_acc = evaluate(logits_eval, loss_eval, train_x, train_y)
  print('Validation error:')
  valid_loss, valid_acc = evaluate(logits_eval, loss_eval, valid_x, valid_y)
  plot_data['train_loss'] += [train_loss]
  plot_data['valid_loss'] += [valid_loss]
  plot_data['train_acc'] += [train_acc]
  plot_data['valid_acc'] += [valid_acc]
  plot_data['lr'] += [lr.eval(session=sess)]
  plot_training_progress(SAVE_DIR, plot_data)
```


### Bonus zadatak - Multiclass hinge-loss


<a name='add'></a>

### Dodatni materijali

- [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io)

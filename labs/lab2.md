---
layout: page
mathjax: true
permalink: /lab2/
---

- [Konvolucijski modeli](#cnn)
- [Vježba](#vjezba)
  - [1. zadatak](#1zad)
  - [2. zadatak](#2zad)
  - [3. zadatak](#3zad)
  - [4. zadatak](#4zad)
- [Dodatni materijali](#add)


<a name='cnn'></a>

## 2. vježba: konvolucijski modeli (CNN)

U ovoj vježbi bavimo se konvolucijskim modelima. 
Ti modeli prikladni su za obradu podataka s topologijom rešetke, 
gdje je osobito važno ostvariti kovarijantnost na translaciju.
<!--(problem miješanja dimenzija u podacima).-->
Dobar primjer takvih podataka su tipične RGB slike 
u kojima se isti objekt može pojaviti na mnogim prostornim lokacijama. 
Potpuno povezani slojevi nisu prikladni za analizu takvih slika
jer svaka aktivacija ovisi o svim pikselima 
i tako od šume ne vidi stabla. 
Takav pristup bi ohrabrivao specijalizaciju aktivacija 
na pojedine dijelove slike, što bi značilo da bi model
morao odvojeno učiti kako jedan te isti objekt
izgleda u različitim dijelovima slike.
Takva situacija pogodovala bi prenaučenosti 
odnosno vodila bi na lošu generalizaciju.
Osim toga dodatni problem je što slike 
tipično sadrže puno piksela. 
Na primjer, prosječne dimenzije slike 
iz poznatog dataseta ImageNet iznose 3x200x200 
što znači da bi svaka aktivacija iz prvog sloja
trebala imati 3\*200\*200=120,000 težina. 
Takva situacija je neodrživa 
jer je broj parametara ograničen GPU memorijom.

Vidimo da bi puno bolje za nas bilo 
kad bi svaka aktivacija bila lokalno povezana 
s malenim susjedstvom prethodnog sloja, 
jer bi to uvelike smanjilo broj težina. 
Aktivacije u ranim slojevima imale bi malo receptivno polje
i mogle bi modelirati samo jednostavne značajke 
koje bi detektirale jednostavne uzorke poput linija i rubova. 
Kasniji slojevi hijerarhijski bi gradili 
sve kompleksnije i kompleksnije značajke
koje bi imale sve veće i veće receptivno polje.
Nadalje, kovarijantnost na translacije unutar slike
mogli bismo ostvariti dijeljenjem parametara 
aktivacija na različitim prostornim lokacijama. 
U takvoj organizaciji, latentne aktivacije sadrže 
izlaz istog podmodela na različitim lokacijama slike. 
Težine konvolucijskih slojeva obično nazivamo filtrima.
Svaki filtar definira afinu transformaciju
malenog lokalnog susjedstva prethodnog sloja 
koja se naknadno ulančava s nelinearnom aktivacijskom funkcijom.
Tipično, lokalno susjedstvo filtra ima kvadratni oblik 
k✕k, k ∈ {3,5,7}, a nelinearnost je zglobnica.
Da zaključimo, konvolucijski modeli koriste tri važne ideje: 
rijetku povezanost, dijeljenje parametara i
ekvivarijantnost reprezentacije.

<div class="fig figcenter fighighlight">
  <img src="/assets/lab2/vv_convnet2.png" width="100%">
  <!--<img src="/assets/lab2/convnet2.png" width="30%">-->
  <div class="figcaption figcenter">Primjer konvolucijskog modela za klasifikaciju slika. 
    Tipično se izmjenjuju konvolucijski slojevi i slojevi sažimanja. 
    Na kraju se dolazi do vektora značajki koji se potpuno povezanim slojem 
    preslikava u konačnu kategoričku distribuciju preko poznatih razreda.
    Slika je preuzeta iz diplomskog rada 
     <a href="http://www.zemris.fer.hr/~ssegvic/project/pubs/vukotic14ms.pdf">Vedrana Vukotića</a>.
  </div>
</div>

<a name='vjezba'></a>

## Vježba

Kod za prva dva zadatka nalazi se 
[ovdje](https://github.com/ivankreso/fer-deep-learning/tree/master/lab2).
Biblioteke koje su vam potrebne za ovu vježbu su 
[PyTorch](http://pytorch.org), 
[torchvision](https://pytorch.org/docs/stable/torchvision/index.html),
NumPy, [Cython](http://cython.org), 
[matplotlib](http://matplotlib.org/) i [scikit-image](http://scikit-image.org/).
Module Pythona najlakše je povući iz pipa.
Druga opcija je koristiti distribucijske pakete. 
Pazite da odaberete verzije za Python 3.

U datoteci `layers.py` nalaze se definicije slojeva 
od kojih se mogu graditi duboki konvolucijski modeli.
Svaki sloj sadrži tri metode potrebne za izvođenje backpropagation algoritma.
Metoda `forward` izvodi unaprijedni prolazak kroz sloj i vraća rezultat.
Metode `backward_inputs` i `backward_params` izvode unatražni prolaz.
Metoda `backward_inputs` računa gradijent s obzirom na ulazne podatke 
(\\( \frac{∂L}{∂\mathbf{x}} \\) gdje je \\(\mathbf{x}\\) ulaz u sloj).
Metoda  `backward_params` računa gradijent s obzirom na parametre sloja (\\( \frac{∂L}{∂\mathbf{w}} \\) gdje vektor \\(\mathbf{w}\\) vektor predstavlja sve parametre sloja)).

<a name='1zad'></a>

### 1. zadatak (25%)
Dovršite implementacije potpuno povezanog sloja, sloja nelinearnosti 
te funkcije gubitka u razredima `FC`, `ReLU` i `SoftmaxCrossEntropyWithLogits`.
Podsjetimo se, gubitak unakrsne entropije računa udaljenost između
točne distribucije i distribucije koju predviđa model i definiran je kao:

$$
L = - \sum_{i=1}^{C} y_i log(s_i(\mathbf{x})) \; .\\
$$

U prikazanoj jednadžbi C predstavlja broj razreda,
a \\( \mathbf{x} \\) ulaz funkcije softmaks
kojeg možemo zvati klasifikacijska mjera ili logit.
Vektor \\( \mathbf{y} \\) sadrži točnu distribuciju 
preko svih razreda za dani primjer.
Tu distribuciju najčešće zadajemo 
jednojediničnim (eng. one-hot) vektorom. 
Vektor \\( s_i(\mathbf{x}) \\) predstavlja 
izlaz funkcije softmax za razred \\(i\\).
Radi jednostavnosti, jednadžba prikazuje 
gubitak za samo jedan primjer,
dok ćemo u praksi obično razmatrati 
prosječan gubitak preko svih primjera mini-grupe.
Da biste izveli unazadni prolazak kroz sloj potrebno je najprije izračunati
gradijent gubitka s obzirom na logite \\( \frac{∂L}{∂\mathbf{x}} \\).
Izvod možemo pojednostavniti tako da 
unaprijed raspišemo funkciju softmax:

$$
log(s_i(x)) = log \left(\frac{e^{x_i}}{\sum_{j=1}^{C} e^{x_j}}\right) = x_i - log \sum_{j=1}^{C} e^{x_j} \\
L = - \sum_{i=1}^{C} y_i \left(x_i - log \sum_{j=1}^{C} e^{x_j}\right) = - \sum_{i=1}^{C} y_i x_i + log \left(\sum_{j=1}^{C} e^{x_j}\right) \sum_{i=1}^{C} y_i \;\; ; \;\;\;\; \sum_{i=1}^{C} y_i = 1 \\
L = log \left(\sum_{j=1}^{C} e^{x_j}\right) - \sum_{i=1}^{C} y_i x_i \\
$$

<!---
\sum_{i=1}^{C} y_i log(s_j(x)) \\
L = log \left(\sum_{j=1}^{C} e^{x_j}\right) - \sum_{i=1}^{C} y_i x_i \\
-->


Sada možemo jednostavno izračunati derivaciju 
funkcije cilja s obzirom na k-ti logit \\( x_k \\):

$$
\frac{∂L}{∂x_k} = \frac{∂}{∂x_k} log \left(\sum_{j=1}^{C} e^{x_j}\right) - \frac{∂}{∂x_k} \sum_{i=1}^{C} y_i x_i \\
\frac{∂}{∂x_k} log \left(\sum_{j=1}^{C} e^{x_j}\right)
  = \frac{1}{\sum_{j=1}^{C} e^{x_j}} \cdot e^{x_k}
  = s_k(\mathbf{x}) \\
\frac{∂L}{∂x_k} = s_k(\mathbf{x}) - y_k \\
$$

Konačno, gradijent s obzirom na sve logite
dobivamo kao vektorsku razliku između  
predikcije modela i točne distribucije:

$$
\frac{∂L}{∂\mathbf{x}} = s(\mathbf{x}) - \mathbf{y} \\
$$

Kako biste bili sigurni da ste ispravno napisali sve slojeve 
testirajte gradijente pozivom skripte `check_grads.py`.
Zadovoljavajuća relativna greška bi trebala biti manja od \\(10^{-5}\\) 
ako vaši tenzori imaju dvostruku preciznost.
Proučite izvorni kod te skripte jer će vam 
ta funkcionalnost biti vrlo korisna za treću vježbu.
Razmislite zašto pri učenju dubokih modela radije koristimo 
analitičke nego numeričke gradijente.

Sada prevedite Cython modul `im2col_cython.pyx` 
pozivom `python3 setup_cython.py build_ext --inplace`
te po potrebi izmijenite varijable `DATA_DIR` i `SAVE_DIR`. 
Proučite izvorni kod funkcija `col2im_cython` i `im2col_cython`
te istražite kao se te funkcije koriste.

Proučite i skicirajte model zadan objektom `net` u skripti `train.py`.
Odredite veličine tenzora te broj parametara u svakom sloju.
Odredite veličinu receptivnog polja značajki 
iz posljednjeg (drugog) konvolucijskog sloja.
Procijenite ukupnu količinu memorije za pohranjivanje aktivacija 
koje su potrebne za provođenje backpropa
ako učimo s mini-grupama od 50 slika.

Napokon, pokrenite učenje modela pozivom skripte `train.py`. 
Odredite vezu između početnog iznosa funkcije gubitka i broja razreda C.
Tijekom učenja možete promatrati vizualizaciju filtara 
koji se spremaju u kazalo `SAVE_DIR`.
Budući da svaka težina odgovara jednom pikselu slike, 
u vašem pregledniku isključite automatsko glađenje slike.
Preporuka je da na Linuxu koristite preglednik Geeqie.

<a name='2zad'></a>

### 2. zadatak (25%)

U ovom zadatku trebate dodati podršku za
L2 regularizaciju parametara.
Dovršite implementaciju sloja `L2Regularizer`
te naučite regularizirani model iz
prethodnog zadatka koji se nalazi u `train_l2reg.py`.
Proučite efekte regularizacijskog hiper-parametra
tako da naučite tri različita modela s
\\( \lambda = 1e^{-3}, \lambda=1e^{-2}, \lambda=1e^{-1} \\)
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
  <img src="/assets/lab2/filters1_big.png" width="80%">
  <img src="/assets/lab2/filters2_big.png" width="80%">
  <div class="figcaption">Slučajno inicijalizirani filtri u prvom sloju na početku učenja (iznad) i naučeni filtri (ispod) s regularizacijom lambda = 0.01.</div>
</div>

<a name='3zad'></a>

### 3. zadatak - usporedba s Tensorflowom (25%)

U Tensorflowu definirajte i naučite model koji je ekvivalentan regulariziranom modelu iz 2. zadatka.
Koristite identičnu arhitekturu i parametre učenja da biste reproducirali rezultate.
Konvoluciju zadajte operacijama `tf.nn.conv2d` ili `tf.contrib.layers.convolution2d`.
U nastavku teksta navodimo primjer korištenja
konvolucije iz paketa `tf.contrib.layers`.

```python
import tensorflow.contrib.layers as layers

def build_model(inputs, labels, num_classes):
  weight_decay = ...
  conv1sz = ...
  fc3sz = ...
  with tf.contrib.framework.arg_scope([layers.convolution2d],
      kernel_size=5, stride=1, padding='SAME', activation_fn=tf.nn.relu,
      weights_initializer=layers.variance_scaling_initializer(),
      weights_regularizer=layers.l2_regularizer(weight_decay)):

    net = layers.convolution2d(inputs, conv1sz, scope='conv1')
    # ostatak konvolucijskih i pooling slojeva
    ...

  with tf.contrib.framework.arg_scope([layers.fully_connected],
      activation_fn=tf.nn.relu,
      weights_initializer=layers.variance_scaling_initializer(),
      weights_regularizer=layers.l2_regularizer(weight_decay)):

    # sada definiramo potpuno povezane slojeve
    # ali najprije prebacimo 4D tenzor u matricu
    net = layers.flatten(inputs)
    net = layers.fully_connected(net, fc3sz, scope='fc3')

  logits = layers.fully_connected(net, num_classes, activation_fn=None, scope='logits')
  loss = ...

  return logits, loss
```

Ako želite koristiti `tf.nn.conv2d` onda će vam od pomoći biti službeni
[tutorial](https://www.tensorflow.org/tutorials/estimators/cnn)
i [dokumentacija](https://www.tensorflow.org/versions/master/api_docs/python/nn.html#convolution).

Tijekom učenja vizualizirajte filtre u prvom sloju kao u prethodnoj vježbi.
Nakon svake epohe učenja pohranite filtre i gubitak u datoteku (ili koristite Tensorboard).
Na kraju učenja prikažite kretanje gubitka kroz epohe (matplotlib).

<!---
Dodajte u model normalizaciju podataka po slojevima nakon svakog konvolucijskog sloja ([Batch
normalization](https://arxiv.org/abs/1502.03167)). To najlakše možete
napraviti tako da konvoluciji zadate
`tf.contrib.layers.batch_norm`
kao parametar normalizacije kako je prikazano ispod:
-->



<a name='4zad'></a>
### 4. zadatak - Klasifikacija na CIFAR-10 skupu (25%)
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset sastoji se od 50000 slika za učenje i validaciju te 10000 slika za
testiranje dimenzija 32x32 podijeljenih u 10 razreda.
Najprije skinite dataset pripremljen za Python [ovdje](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).
Iskorisite sljedeći kod kako biste učitali podatke i pripremili ih.

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

train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32).transpose(0,2,3,1)
train_y = []
for i in range(1, 6):
  subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
  train_x = np.vstack((train_x, subset['data']))
  train_y += subset['labels']
train_x = train_x.reshape((-1, num_channels, img_height, img_width))
train_y = np.array(train_y, dtype=np.int32)

subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0,2,3,1).astype(np.float32)
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

train_x = train_x.transpose(0,3,1,2)
valid_x = valid_x.transpose(0,3,1,2)
test_x = test_x.transpose(0,3,1,2)
```

Vaš zadatak je da u Pytorch-u naučite CNN na ovom skupu.
U nastavku je prijedlog jednostavnog modela 
s kojom biste trebali dobiti ukupnu točnost oko 70% na validacijskom skupu:

```
conv(16,5) -> relu() -> pool(3,2) -> conv(32,5) -> relu() -> pool(3,2) -> fc(256) -> relu() -> fc(128) -> relu() -> fc(10)
```
gdje `conv(16,5)` predstavlja konvoluciju sa 16 mapa te dimenzijom filtra 5x5,
a `pool(3,2)` max-pooling sloj s oknom veličine 3x3 i pomakom (*stride*) 2.
Prilikom treniranja padajuću stopu učenja možete implementirati korištenjem
`torch.optim.lr_scheduler.ExponentialLR`

Napišite funkciju `evaluate` koja na temelju predviđenih i točnih indeksa razreda određuje pokazatelje klasifikacijske performanse:
ukupnu točnost klasifikacije, matricu zabune (engl. confusion matrix) u kojoj retci odgovaraju točnim razredima a stupci predikcijama te mjere preciznosti
i odziva pojedinih razreda. U implementaciji prvo izračunajte matricu zabune, a onda sve ostale pokazatelje na temelju nje.
Tijekom učenja pozivajte funkciju `evaluate` nakon svake epohe na skupu za učenje i
validacijskom skupu te na grafu pratite sljedeće vrijednosti: prosječnu vrijednost
funkcije gubitka, stopu učenja te ukupnu točnost klasifikacije.
Preporuka je da funkciji provedete samo
samo unaprijedni prolazak kroz dane primjere koristeći `torch.no_grad()` i pritom izracunati matricu zabune.
Pazite da slučajno ne pozovete i operaciju koja provodi učenje tijekom evaluacije.
Na kraju funkcije možete izračunati ostale pokazatelje te ih isprintati.

<div class="fig figcenter fighighlight">
  <img src="/assets/lab2/training_plot.png" width="100%">
  <div class="figcaption figcenter">Primjer grafa učenja za nvedeni model uz veličinu grupe od 50</div>
</div>

Vizualizirajte slučajno inicijalizirane
Težine konvolucijskog sloja možeze dohvatiti
korištenjem 
`conv.weights`.
U nastavku je primjer kako to može izgledati, ovisno
o načinu implementiranj konvolucijske mreže.

```python
net = ConvNet()

draw_conv_filters(0, 0, net.conv1.weight.detach().numpy(), SAVE_DIR)
```

U nastavku se nalazi kod koji možete
koristiti za vizualizaciju:

```python
def draw_conv_filters(epoch, step, weights, save_dir):
  w = weights.copy()
  num_filters = w.shape[0]
  num_channels = w.shape[1]
  k = w.shape[2]
  assert w.shape[3] == w.shape[2]
  w = w.transpose(2,3,1,0)
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
  <div class="figcaption figcenter">CIFAR-10: slučajno inicijalizirani filtri u prvom sloju na početku učenja (iznad) i naučeni filtri (ispod) s regularizacijom lambda = 0.0001.</div>
</div>

Prikažite 20 netočno klasificiranih slika s najvećim gubitkom te ispišite njihov točan razred,
kao i top-3 razreda za koje je model dao najveću vjerojatnost.
Da biste prikazali sliku, morate najprije poništiti normalizaciju srednje vrijednosti i
varijance:

```python
import skimage as ski
import skimage.io

def draw_image(img, mean, std):
  img = img.transposse(1,2,0)
  img *= std
  img += mean
  img = img.astype(np.uint8)
  ski.io.imshow(img)
  ski.io.show()
```

Ispod se nalazi kod koji možete iskoristiti za crtanje grafova:

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
        # broj primjera djeljiv bsz
        batch_X = X[batch*bsz:(batch+1)*bsz, :]
        batch_Yoh = Yoh[batch*bsz:(batch+1)*bsz, :]

        loss = model.get_loss(batch_X, batch_Yoh)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch%100==0:
            print("epoch: {}, step: {}/{}, batch_loss: {}".format(epoch, batch, n_batch, loss))

        if batch%200==0:
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

Ukoliko imate GPU, možda će vam biti zanimljivo pokušati dobiti bolje rezultate s moćnijom
arhitekturom. U tom slučaju [ovdje](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130)
možete pronaći pregled članaka koji imaju najbolje rezultate na ovom skupu.
Kao što vidite trenutni *state of the art* je oko 96% ukupne točnosti.
Dva važna trika koje koriste najbolje arhitekture su skaliranje slika na veću rezoluciju
kako bi omogućili da prvi konvolucijski slojevi uče značajke jako niske razine
te proširivanje skupa za učenje raznim modificiranjem slika (*data jittering*).
Bez ovih trikova je jako teško preći preko 90% ukupne točnosti.


### Bonus zadatak - Multiclass hinge loss (max 20%)

Ovaj zadatak razmatra učenje modela za slike iz CIFARA
s alternativnom formulacijom gubitka
koju nismo obradili na predavanjima.
Cilj je zamijeniti unakrsnu entropiju 
višerazrednom inačicom gubitka zglobnice.
Objašnjenje tog gubitka možete pronaći 
[ovdje](http://cs231n.github.io/linear-classify/#svm).
Za sve bodove zadatak je potrebno riješiti
primjenom osnovnih Pytorch operacija nad tenzorima
te usporediti postignute rezultate.

Pomoć: sučelje nove funkcije gubitka
moglo bi izgledati ovako:
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
Rješenje možete započeti razdvajanjem 
izlaza posljednjeg potpuno povezanog sloja
na vektor logita točnih razreda
i matricu logita netočnih razreda.
To možete provesti pozivom funkcije
[torch.masked_select](https://pytorch.org/docs/stable/torch.html#torch.masked_select)
pri čemu masku zadajete regularnom odnosno invertiranom 
verzijom matrice s jednojediničnim oznakama podataka.
Sada razliku između matrice logita netočnih razreda
i vektora logita točnih razreda
možemo izračunati običnim oduzimanjem,
jer Pytorch automatski umnaža (eng. broadcast) operand nižeg reda.
Pripazite da sve tenzore preoblikujete na ispravni oblik,
jer funkcija `torch.masked_select` vraća tenzor prvog reda.
Maksimum po elementima možete računati 
odgovarajućom varijantom funkcije 
[torch.max](https://pytorch.org/docs/stable/torch.html#torch.max).

<!--
  [torch.nn.functional.one_hot](https://pytorch.org/docs/stable/nn.functional.html#one-hot)
 -->
<!--
Pokušajte u zadnjem zadatku unakrsnu entropiju zamijeniti 
višerazrednim gubitkom zglobnice te usporedite rezultate. 
Objašnjenje tog gubitka možete pronaći [ovdje](http://cs231n.github.io/linear-classify/#svm).
Za sve bodove zadatak je potrebno ostvariti 
primjenom osnovnih Tensorflowovih operacija nad tenzorima. 

Pomoć: jedna opcija kako to možete izvesti je da razdvojite logite
(izlazi iz zadnjeg potpuno povezanog sloja) na matricu logita netočnih razreda i vektor
logita na mjestima točnih razreda.
To možete izvesti pomoću operacija `tf.dynamic_partition` i `tf.one_hot`.
Zatim unutar `tf.maximum` računate razliku 
između matrice logita na netočnim razredima 
i vektora logita na točnim razredima. 
To možete napisati kao običnu razliku 
jer za tenzore različitih dimenzija 
Tensorflow po defaultu napravi *broadcasting* 
ako je to moguće.
-->

<a name='add'></a>

### Dodatni materijali

- [Deep learning book](http://www.deeplearningbook.org/contents/convnets.html)
- [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io)

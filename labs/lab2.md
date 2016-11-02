---
layout: page
mathjax: true
permalink: /lab2/
---

- [Konvolucijska neuronska mreža](#cnn)
- [Vježba](#vjezba)
  - [1. zadatak](#zad1)
  - [2. zadatak](#zad2)
  - [3. zadatak](#zad3)
  - [4. zadatak](#zad4)
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
Neurone u konvolucijskim slojevima mreže obično nazivamo i filteri.
Konvolucijske mreže koriste tri važne ideje: rijetku povezanost, dijeljenje parametara i
ekvivarijantnost reprezentacije.


**Convolution Demo**. Below is a running demo of a CONV layer. Since 3D volumes are hard to visualize, all the volumes (the input volume (in blue), the weight volumes (in red), the output volume (in green)) are visualized with each depth slice stacked in rows. The input volume is of size \\(W_1 = 5, H_1 = 5, D_1 = 3\\), and the CONV layer parameters are \\(K = 2, F = 3, S = 2, P = 1\\). That is, we have two filters of size \\(3 \times 3\\), and they are applied with a stride of 2. Therefore, the output volume size has spatial size (5 - 3 + 2)/2 + 1 = 3. Moreover, notice that a padding of \\(P = 1\\) is applied to the input volume, making the outer border of the input volume zero. The visualization below iterates over the output activations (green), and shows that each element is computed by elementwise multiplying the highlighted input (blue) with the filter (red), summing it up, and then offsetting the result by the bias.

<div class="fig figcenter fighighlight">
  <iframe src="/assets/conv-demo/index.html" width="100%" height="700px;" style="border:none;"></iframe>
  <div class="figcaption"></div>
</div>

- izvod backward pass za l2 regularizacija, relu, cross entropy

<a name='vjezba'></a>

## Vježba

Kod za prva dva zadatka nalazi se [ovdje](https://github.com/dlunizg/lab2).
U datoteci `layers.py` izvedeni su osnovni slojevi od kojih se sastoji CNN.
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

Kako biste bili sigurni da je kod svih slojeva ispravan testirajte gradijente pozivom skripte `check_grads.py`.
Zadovoljavajuća relativna greška bi trebala biti manja od \\(10^{-5}\\) ako vaši tenzori imaju dvostruku preciznost.
Napokon, pokrenite učenje modela pozivom skripte `train.py`. Napomena: najprije postavite odgovarajuće puteve u varijable
`DATA_DIR` i `SAVE_DIR`;

<a name='2zad'></a>

### 2. zadatak
Dovršite implementaciju `L2Regularizer` i `RegularizedLoss` slojeva.


Dodajte L2 regularizaciju konvolucijskom i potpuno povezanom sloju.
Minimalno je potrebno izmijeniti update\_params metode.
Dodajte novi sloj L2Loss koji na ulazu forward metode prima tenzor težine te
implementirajte odgovarajući unaprijedni i unazadni prolazak.
Dodajte novi sloj MultiLoss koji na ulaz prima listu funkcija cilja te računa njihov zbroj.
Konačna funkcija cilja je sada MultiLoss koji u našem slučaju na ulazu treba dobiti
listu u kojoj se nalazi funkcija cilja unakrsne entropije i L2 regularizacije težina
konvolucijskih i potpuno povezanih slojeva.


<a name='3zad'></a>

### 3. zadatak Usporedba s Tensorflow
U Tensorflowu definirajte i naučite model koji je ekvivalentan modelu iz 2. zadatka.
Korisite identičnu arhitekturu i parametre učenja.
Kako biste u graf dodali operaciju konvolucije koristite `tf.nn.conv2d` ili `tf.contrib.layers.convolution2d`.
Prije toga proučite dio dokumentacije vezan za konvoluciju https://www.tensorflow.org/versions/master/api_docs/python/nn.html#convolution.

- dodajte jos jedan sloj
- usporedite s i bez BN
- dodajte


<a name='4zad'></a>

### 4. zadatak - CIFAR-10

Napišite funkciju `eval(Y,Y_)` koja na temelju predviđenih i točnih indeksa razreda određuje pokazatelje klasifikacijske performanse:
ukupnu točnost klasifikacije, matricu zabune (engl. confusion matrix) u kojoj retci odgovaraju točnim razredima a stupci predikcijama te vektore preciznosti
i odziva pojedinih razreda. U implementaciji prvo izračunajte matricu zabune, a onda sve ostale pokazatelje na temelju nje.

- vizualizirajte filtre u prvom sloju
- plotajte train validation loss i accuracy po epohi
- prikažite N slučajno odabranih netočno klasificiranih slika i ispod ispisite top 3 klase
  te vjerojtnosti koje su im pridjeljene.
- provjerite graf u tensofboardu?



### Bonus zadatak - Multiclass hinge-loss


<a name='add'></a>

### Dodatni materijali

- [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io)

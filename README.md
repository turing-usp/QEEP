# QEEP Turing (Qual é esse Pokémon?)
<img src="https://i.imgur.com/NAXImVj.jpg" width="1000">
<img src="https://img.shields.io/github/contributors/GrupoTuring/QEEP"> <img src="https://img.shields.io/github/last-commit/GrupoTuring/QEEP">

## O que é o QEEP?
O QEEP Turing (Qual é esse Pokémon?) é um projeto desenvolvido pelos membros da área de Visão Computacional do grupo Turing para fazer a detecção de pokémons em imagens. O projeto começou com o objetivo de incentivar os membros a estudar e desenvolver, de maneira prática, técnicas de detecção de objetos.

Para isso, membros fizeram um classificador de pokémons utilizando duas redes distintas: [ShuffleNet](https://towardsdatascience.com/review-shufflenet-v1-light-weight-model-image-classification-5b253dfe982f) e [MobileNet](https://towardsdatascience.com/review-mobilenetv1-depthwise-separable-convolution-light-weight-model-a382df364b69). Em seguida foi criado um detector utilizando o algoritmo de [Sliding Windows](https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/). 

Para clonar o repositório digitar:

```bash
$ git clone https://github.com/GrupoTuring/QEEP.git
```

## Requisitos
Python 3.8 ou versões superiores e as dependências contidas no arquivo [`requiments.txt`](https://github.com/GrupoTuring/QEEP/blob/master/requirements.txt). Para instalá-las basta executar:
```bash
$ pip install -r requirements.txt
```

## Classificador
O detector pode ser utilizado com 2 redes: MobileNet e ShuffleNet. Uma breve explicação de cada uma é apresentada abaixo.

### MobileNet
<img src="https://miro.medium.com/max/2414/1*RpIVS4iOLyNDVeTfFi2biw@2x.png" width="800">
MobileNet é uma simplificação de redes neurais para possibilitar o seu uso em aplicações web, com isso podemos criar rapidamente uma aplicação de reconhecimento de images. É um modelo de CNN que implementa a chamada depthwise separable convolution, um recurso utilizado partindo do princípio da separação entre os dimensões de tensores (separação entre altura/largura e profundidade).

Nesse repositório foi feito um [transfer learning](https://medium.com/turing-talks/deep-transfer-learning-a145125b754c) com tal rede utilizando a implementação padrão do PyTorch que pode ser [conferida aqui](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/).


### ShuffleNet
<img src="https://miro.medium.com/max/1112/1*4YsmTx-vhYISZRFu7422lQ.png" width="800">
A ShuffleNet é uma rede neural projetada especialmente para dispositivos móveis com poder de computação muito limitado. A arquitetura utiliza duas novas operações, convolução de grupo pontual e troca de canais, para reduzir significativamente o custo de computação, mantendo a precisão.

Assim como na MobileNet, foi feito um [transfer learning](https://medium.com/turing-talks/deep-transfer-learning-a145125b754c) com tal rede utilizando a implementação padrão do PyTorch que pode ser [conferida aqui](https://pytorch.org/hub/pytorch_vision_shufflenet_v2/).

### Comparativo
Ambas as redes foram escolhidas visando eficiência e praticidade na hora da predição, podendo ser facilmente utilizadas numa aplicação web ou mobile. Um comparativo de tempo entre as redes foi elaborado.

| Model | 1 imagem | 10 imagens | 100 imagens | 1000 imagens | 
|---------- |------ |------ | ------ | :-------: |
| [MobileNet](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/) | 43.3   | 43.3   | 63.0  | 4.8ms     |
| [ShuffleNet](https://pytorch.org/hub/pytorch_vision_shufflenet_v2/)  | 44.3   | 44.3   | 64.6  | 4.9ms     |



## Inferência

## Treinamento

## Sobre Nós
O Grupo Turing é o grupo de extensão acadêmica da Universidade de São Paulo que estuda, dissemina e aplica conhecimentos de Inteligência Artificial. Surgiu em 2015 como um grupo de estudos originalmente idealizado por duas mulheres, fundado por um grupo de três politécnicos e batizado em homenagem a Alan Turing (1912-1954), matemático e lógico inglês considerado o pai da computação.

## Contate-nos
Caso queira conhecer melhor o Grupo Turing podem nos acompanhar em nossas redes sociais:

- [Facebook](https://www.facebook.com/grupoturing.usp)
- [Linkedin](https://www.linkedin.com/company/grupo-turing/)
- [Instagram](https://www.instagram.com/grupoturing.usp/)
- [Medium](https://medium.com/turing-talks)
- [Discord](https://discord.com/invite/26RGmBS)

<img src="https://i.ibb.co/DtHQ3FG/802x265-Logo-GT.png" width="400"> 

## Grupo Turing
# Visão Computacional: Localização de Objetos
#### Por: [Eduardo Eiras](https://github.com/dueiras);  [Guilherme Salustiano](https://github.com/guissalustiano); [Luis Santos](https://github.com/luizsantos-1); [Noel Eliezer](https://github.com/anor4k); [Rafael Coelho](https://github.com/rafael-acoelho); [Rodrigo Estevam](https://github.com/materloki); [Wesley de Almeida](https://github.com/WesPereira)
Projeto da área Visão Computacional com o foco de fazer uma rede classificadora de objetos e a utilizar para criar um localizador de objetos.

## Qual a diferença ?

- Um classificador de objetos recebe uma imagem e tem como output qual classe aquela imagem pertence.
- Um localizador de objetos retorna onde em uma imagem contém uma determinada classe.

## Treinando o modelo
Para realizar o _transfer learning_ foi utilizado o modelo da YOLOv3 da [ultralytics](https://github.com/ultralytics/yolov3). Para isso, o dataset deve seguir a estrutura descrita a seguir. A estrutura dos arquivo deve ser a seguinte:
```
Pasta do modelo
--- data (pasta)
    --- nome do dataset (pasta)
        --- imagens (pasta)
            --- img1.jpg
            --- img2.jpg
            ..........
        --- rotulos (pasta)
            --- img1.txt
            --- img2.txt
            ..........
        --- treino.txt
        --- validacao.txt
```
Os arquivos <code>img*.txt</code> são arquivos que seguem a seguinte estrutura:
```
0 x_center y_center img_width img_height
0 0.500968 0.550968 0.1203843 0.4503683
0 0.500968 0.550968 0.1203843 0.4503683
0 0.500968 0.550968 0.1203843 0.4503683
...
```
Nesses arquivos, o primeiro parâmetro representa a classe do objeto que será classificado seguido da posição do centro dele (x_center, y_center). Em seguida, temos a largura e altura da caixa que envolve o objeto a ser classificado. **Vale ressaltar que todas as medidas são relativas ao tamanho da imagem a ser classificada**. Cada linha do arquivo contém um dos objetos a ser classificado na imagem.<br>
Os arquivos <code>treino.txt</code> e <code>validacao.txt</code> contém o caminho relativo (considerando a raiz do projeto) para as imagens. Veja:
```
../data/dataset/imagens/img1.jpg
../data/dataset/imagens/img2.jpg
...
```
class ModelDataset():
    """
    Classe basica para manipular modelos
    """
    model: torch.nn.Module
    dataloader: torch.utils.data.DataLoader

    def load_dataset(self, path: str = "./data", tranform: torch.nn.Module = None):
            """
            Descrição
            --------
            Carrega o Dataset

            Entradas
            --------
            path: str
            Diretorio que será montado as classes na seguinte extrutura
            <path>
            ├── bulbassauro
            │   ├── imagem1.png
            │   ├── imagem2.png
            │   ├── ...
            │   └── imagemN.py
            ├── pikachu
            │   ├── imagem1.png
            │   ├── imagem2.png
            │   ├── ...
            │   └── imagemN.py
            ...

            tranform: torch Tranform
            Transformaçoes a serem aplicadas no dataset
            Se não for defenido será usado o default_tranform
            """
            if not Path(path).exists():
                raise Exception('Dataset not found')

            if tranform is None:
                tranform = default_transform

            self.dataset = datasets.ImageFolder(root=path,
                                                transform=tranform)
            self.dataset_classes = self.dataset.classes

        def split_dataset(self, tresh_hold: float = 0.8, path: str = ".data", transform: torch.nn.Module = None):
            """
            Descrição
            --------
            Separa o dataset carregado em dois grupos dividido pelo tresh_hold
            obs: precisa ter o datasetCarregado

            Entradas
            --------
            tresh_hold: float
            Porcentagem de treino em relação ao dataset original

            Saídas
            ------
            torch Dataset
            Dataset de treino

            torch Dataset
            Dataset de validação

            torch Dataset
            Dataset completo
            """
            self.dataset_splited = torch.utils.data.random_split(
                self.dataset, [int(len(self.dataset) * tresh_hold), int(len(self.dataset) * (1-tresh_hold))])

    def loadSplitedLoader(batch_size: int = 4, num_workers: int = 4):
        """
        Descrição
        --------
        Carrega o Dataset e separa ele em dois grupos: de treino e validação e os transforma em bachs

        Entradas
        --------
        batch_size: int
        Tamanho de cada batch

        num_workers: int
        Quantidade de subprocessos
        """

        return [torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True, num_workers=num_workers) 
                for d in self.dataset_splited]


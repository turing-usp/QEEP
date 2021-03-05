import gdown
import torch
import torch.nn as nn
import torchvision.models as models
import os
from typing import Type

def loadModel(model : str = "mobilenet", file : str = "mobilenet_weights.pkl", drive : bool = True,
              url : str ="https://drive.google.com/uc?export=download&id=1yC0qK0gVX5sc6GTpBPBupH3TSFssLkqS") -> nn.Module:
    """
    Descrição
    --------
    Carrega uma rede a ser utilizada como modelo
    
    Entradas
    --------
    model: (String)
    O nome do modelo a ser carregado, pode ser:
    'mobilenet' ou 'shufflenet'
    
    file: (Path)
    Caso drive = False, representa o caminho para o
    arquivo onde o modelo será carregado.
    Caso drive = True, representa o caminho para onde
    o modelo será salvo.
    
    drive: (Bool)
    Define se o modelos será carregado do Google Drive 
    ou localmente

    url: (Url)
    Endereço de onde o arquivo deverá ser baixado, caso
    drive = True.
    
    Saídas
    ------
    model_ft: (torch model)
    O modelo que gostaria de ser carregado
    
    """ 
    if drive:
        try:
            gdown.download(url, file, quiet=False)
        except Exception as err:
            print(str(err))
            print("Rede pré treinada não pode ser baixada.")
    
    if model == "mobilenet":
        model_ft = models.mobilenet_v2()
        model_ft.fc = nn.Linear(1000, 151)
    else:
       model_ft = models.shufflenet_v2_x1_0()
       num_feat = model_ft.fc.in_features
       model_ft.fc = nn.Linear(num_feat, 151)
    st = torch.load(file, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    model_ft.load_state_dict(st)
    print(model_ft.eval())
    return model_ft

import torch


def saveModel(name: str, model: torch.nn.Module = None, path: str = "") -> Type[None]:
    """
    Salva um PyTorch model, no path e nome desejados, com extensão .pkl
    ---------------
    Argumentos:
        - model: PyTorch model (torch.nn.Module) 
        - path: string contendo o path para salvar o modelo
        - name: string contendo o nome do arquivo salvo
    """
    # Ajustando o path
    PATH = os.path.join(path, name + '.pkl')
    torch.save(model.state_dict(), PATH)
    
if __name__ == "__main__":
    model = loadModel(model = "mobilenet", drive=True)

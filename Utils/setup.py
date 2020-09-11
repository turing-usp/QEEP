from setuptools import setup

setup(name='Augmentation',
      version='0.0.1',
      description='Pacote para criar imagens augmentadas para treinamento de uma rede de localização de objetos', 
      author='Grupo Turing',
      author_email='turing.usp@gmail.com',
      packages=['Augmentation'],
      install_requires=['matplotlib', 
                        'numpy',
                        'pillow',
                        'opencv'])
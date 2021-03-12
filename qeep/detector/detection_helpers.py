# import the necessary packages
import imutils
import numpy as np
import PIL as Image

def sliding_window(image: Image, step: int, ws: tuple) -> Image:
	"""Passa a janela deslizante na imagem.
	# Entradas
		image: PIL Image.
		step: passos, em pixels, da janela deslizante
		ws: formato da janela
	# Saidas
		A janela atual
	"""
	# passa a janela pela imagem
	for y in range(0, image.shape[0] - ws[1], step):
		for x in range(0, image.shape[1] - ws[0], step):
			# yield a janela atual
			yield (x, y, image[y:y + ws[1], x:x + ws[0]])

def image_pyramid(image: Image, scale:float =1.5, minSize:tuple=(224, 224))-> Image:
	"""Aplica o algoritmo de piramide na imagem
	# Entradas
		image: PIL Image.
		scale: fator de redução da escala.
		minsize: tamanho minimo da imagem de saida, serve como ponto de parada
	# Saidas
		A imagem para passar as janelas deslizantes
	"""
	# yield a imagem original
	yield image
	# loop para pegar todas as imagens da piramide
	while True:
		# computa o tamanho da proxima imagem
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		# condição de parada: se a imagem ficar menor que o tamanho minimo permitido
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		# yield the next image in the pyramid
		yield image

def img_to_array(img: Image) -> np.array:
	"""Converte uma PIL Image em um Numpy array.
	# Entradas
		img: PIL Image.
	# Saidas
		x: A imagem em Numpy array
	"""
	x = np.asarray(img, dtype="int8")
	if len(x.shape) == 2:
		x = x.reshape((x.shape[0], x.shape[1], 1))
	elif len(x.shape) != 3:
		raise ValueError('Unsupported image shape: %s' % (x.shape,))
	return x


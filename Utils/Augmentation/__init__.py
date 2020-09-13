from PIL import Image, ImageOps
import cv2
import numpy as np
import matplotlib.pyplot as plt

def changeImageSize(maxWidth, maxHeight, image):
    """
    Descrição
    --------
    Muda o tamanho de uma imagem, dado uma altura máxima e largura máxima
    
    Entradas
    --------
    image: (PIL Image)
    Imagem em Pillow em que terá o tamanho mudado
    
    maxWidth: (int)
    Largura máxima para a nova imagem
    
    maxHeight: (int)
    Altura máxima para a nova imagem
    
    Saídas
    ------
    image: (PIL Image)
    Imagem em Pillow com o tamanho mudado

    
    """   
    
    widthRatio  = maxWidth/image.size[0]
    heightRatio = maxHeight/image.size[1]

    newWidth    = int(widthRatio*image.size[0])
    newHeight   = int(heightRatio*image.size[1])

    newImage    = image.resize((newWidth, newHeight))
    return newImage

def mirror(image):
    """
    Descrição
    --------
    Devolve a imagem de entrada espelhada horizontalmente
    
    Entradas
    --------
    image: (PIL Image)
    Imagem em Pillow a ser espelhada
    
    Saídas
    ------
    image: (PIL Image)
    Imagem em Pillow espelhada

    
    """

    return ImageOps.mirror(image)

def rotate(image, angle):
    """
    Descrição
    --------
    Aplica uma rotação na imagem em 'angle' graus
    
    Entradas
    --------
    image: (PIL Image)
    Imagem em Pillow a ser rotacionada
    
    angle: (float)
    Ângulo em que a imagem será rotacionada
    
    Saídas
    ------
    output(type)    image: (PIL Image)
    Imagem em Pillow rotacionada no ângulo especificado
    
    """  
    color = image.getpixel((0,0))
    return image.rotate(angle, Image.NEAREST, expand=1, fillcolor=color)

def randomCrop(img, width, height):
    """
    Descrição
    --------
    Corta aleatoriamente uma imagem, com altura e largura máximas especificadas.
    Assim, a imagem será cortada aleatoriamente entre a posição 0 e sua altura menos 'height',
    e entre 0 e sua largura menos 'width'
    
    Entradas
    --------
    img: (PIL Image)
    Imagem em Pillow a ser recortada
    
    width: (int)
    Largura máxima
    
    height: (int)
    Altura máxima
    
    Saídas
    ------
    data(PIL Image)
    Imagem Pillow convertida de um numpy array já com o corte

    
    """  
    data = np.asarray(img)
    x = np.random.randint(0, img.size[0] - width)
    y = np.random.randint(0, img.size[1] - height)
    data = data[y:y+height, x:x+width]
    return Image.fromarray(data)

def randomObjAugment(image, prob):
    """
    Descrição
    --------
    Funções de augmentation a serem aplicadas aleatoriamente em uma imagem de um objeto
    
    Entradas
    --------
    image: (PIL Image)
    Imagem em formato Pillow 
    
    prob: (float)
    A probabilidade de se aplicar uma rotação aleatória na imagem
    
    Saídas
    ------
    imagem: (PIL Image)
    Imagem com as alterações
    
    """      
    if np.random.random() < 0.5:
        image = mirror(image)
        
    if np.random.random() < prob:
        degree = np.random.randint(low=-10, high=10, size=1)[0]
        image = rotate(image, degree)
        
    maxSize = np.random.randint(low=100, high=450, size=1)[0]
    
    image = changeImageSize(maxSize, maxSize, image)
        
    return image

def makeMask(image):
    """
    Descrição
    --------
    Função que cria uma máscara em uma imagem se baseando apenas na cor da borda.
    É feita uma transformação para passar a imagem para preto e branco e é retirado
    da imagem as partes brancas, que se está sendo suposto ser o fundo a ser retirado.
    
    Entradas
    --------
    image: (PIL Image)
    
    Saídas
    ------
    img: (PIL Image)
    Uma imagem em Pillow convertida de um numpy array
    
    """  
    BW = image.convert('L')
    img = np.asarray(BW).copy()
    img[img==255] = 0
    img[img>0] = 255
    return Image.fromarray(img)

def makeMaskP(image):
    """
    Descrição
    --------
    Função que cria uma máscara em uma imagem modo P se baseando apenas no parâmetro de transparência.
    É feita uma transformação para passar a imagem para preto e branco e são retiradas as partes
    transparentes da imagem.
    
    Entradas
    --------
    image: (PIL Image)
    
    Saídas
    ------
    img: (PIL Image)
    Uma imagem em Pillow convertida de um numpy array
    
    """
    transparency = image.info.get('transparency')
    imgArray = np.asarray(image.copy())
    mask = np.asarray(image.convert('L')).copy()
    mask[imgArray == transparency] = 0
    mask[imgArray != transparency] = 255
    return Image.fromarray(mask)

def randomEnvAugment(image, prob):
    """
    Descrição
    --------
    Funções de augmentation a serem aplicadas aleatoriamente em uma imagem de fundo
    
    Entradas
    --------
    image: (PIL Image)
    Imagem em formato Pillow 
    
    Saídas
    ------
    imagem: (PIL Image)
    Imagem com as alterações
    
    """      
    if np.random.random() < 0.5:
        image = mirror(image)
        
    if np.random.random() < prob:
        degree = np.random.randint(low=-10, high=10, size=1)[0]
        color = image.getpixel((image.size[0]//2,image.size[1]//2))
        image = image.rotate(degree, Image.NEAREST, fillcolor=color)
        
    maxWidth = image.size[0]
    maxHeight = image.size[1]
    
    if np.random.random() < prob:
        ratio = np.random.randint(low=70, high=99, size=1)[0]/100
        width = int(maxWidth * ratio)
        height = int(maxHeight * ratio)
        image = randomCrop(image, width, height)
    
    image = changeImageSize(maxWidth, maxHeight, image)
        
    return image

def randomContrast(image, change_limits):
    """
    Descrição
    --------
    Descrição da função
    
    Entradas
    --------
    input: (type)
    
    Saídas
    ------
    output(type)

    
    """      
    alpha = np.random.random()
    alpha = (alpha*change_limits) - 0.5 * change_limits + 1
    
    new_image = np.zeros_like(image)
    new_image = np.clip(alpha*image, 0, 255)
    new_image = np.array(new_image , dtype= np.int32)
    return new_image

def randomBrightness(image, change_limits):
    """
    Descrição
    --------
    Descrição da função
    
    Entradas
    --------
    input: (type)
    
    Saídas
    ------
    output(type)

    
    """  
    a,b = change_limits
    beta = np.random.randint(a,b)
    
    new_image = np.zeros_like (image)
    new_image = np.clip(image + beta, 0, 255)
    new_image = np.array(new_image , dtype= np.int32)
    return (new_image)

def randomPlace(img, IMG, mask):
    """
    Descrição
    --------
    Insere uma imagem em uma segunda imagem em local aleatório
    
    Entradas
    --------
    img: (PIL Image)
    Imagem em formato Pillow contendo o objeto a ser adicionado ao fundo

    IMG: (PIL Image)
    Imagem em formato Pillow contendo a paisagem de fundo
    
    mask: (numpy array)
    Máscara em array numpy para cortar o fundo da imagem objeto a ser adicionada
    
    Saídas
    ------
    IMG_copy: (PIL Image)
    A imagem de fundo modificada com a imagem objeto
    
    rect: (tuple)
    Uma tupla contendo a posição do canto inferior esquerdo de onde a imagem foi adicionada e as dimensões da imagem adicionada
    
    """  

    w, h = img.size[0], img.size[1]
    W, H = IMG.size[0], IMG.size[1]
    
    x = np.random.randint(0, W-w)
    y = np.random.randint(0, H-h)
    box = (x, y)
    # Como o método "paste" altera diretamente a imagem, 
    # criamos uma cópia dela
    IMG_copy = IMG.copy()
    IMG_copy.paste(img, box, mask)
    
    return IMG_copy, (x, y, w, h)

def blendImages(objImage, envImage, num_images, path):
    """
    Descrição
    --------
    Junta uma imagem objeto em uma imagem de fundo aleatoriamente, aplicando funções de augmentation
    
    Entradas
    --------
    objImage: (PIL image)
    Imagem em formato Pillow contendo o objeto a ser adicionado ao fundo

    envImage: (PIL image)
    Imagem em formato Pillow contendo a paisagem de fundo
    
    num_images: (int)
    Número de imagens a serem criadas
    
    path: (str)
    Caminho do diretório em que serão salvas as novas imagens
    
    Saídas
    ------
    Nenhuma

    
    """    
    prob = 0.7
    for i in range(num_images):
        augObjImage = randomObjAugment(objImage, prob)
        augEnvImage = randomEnvAugment(envImage, prob)
        
        newObjImage = augObjImage
        newEnvImage = augEnvImage
        
        mode = newObjImage.mode
        if mode == 'RGBA':
            mask = newObjImage
        elif mode == 'P':
            mask = makeMaskP(newObjImage)
        else:
            mask = makeMask(newObjImage)
        
        img, rect = randomPlace(newObjImage, newEnvImage, mask)
        
        save_path = path+"/"+str(i+1)+".jpg"
        img.save(save_path)
        
        plt.imshow(img),plt.colorbar(),plt.show()

def applyMask(img_path, n_its=5, step=1, img_show=False):
    """
    Descrição
    ---------
    Aplicar o algoritmo de GrabCut implementado pela OpenCV
    
    Entradas
    --------
    img_path: (str)
    O caminho da imagem
    
    n_its: (int) Default=5
    Número de iterações a serem executadas pelo algoritmo. Cinco se não especificado
    
    step: (int) Default=1
    Número de pixels a serem cortados da borda para criar o retângulo da máscara. Um quando não especificado
    
    img_show: (bool)
    Booleano para mostrar ou não a imagem após aplicação do algoritmo
    
    Saídas
    ------
    img_masked: (cv2 img object)
    Devolve uma imagem de objeto da OpenCV
    
    """
    img = cv2.imread(img_path)
    max_size = 500
    if img.shape[0] >= max_size or img.shape[1] >= max_size:
        h = img.shape[0]
        w = img.shape[1]
        r = h / w
        img = cv2.resize(img, (max_size, int(r*max_size))) # (w, h)
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (0, 0, img.shape[1]-step, img.shape[0]-step) # (x, y, w, h)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, n_its, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_masked = img*mask2[:,:,np.newaxis]
    if img_show:
        plt.imshow(img_masked)
        plt.colorbar()
        plt.show()    
    return img_masked


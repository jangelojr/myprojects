import cv2
import numpy as np
from keras.models import load_model

# Detector de face
face_cascade = cv2.CascadeClassifier('DADOS/haarcascades/haarcascade_frontalface_default.xml')

class ExpressaoFacial:
    def __init__(self, imagem_carregada):
        
        self.imagem_carregada = imagem_carregada 
    
        def detectar_roi(url_da_imagem):
            imagem = cv2.imread(url_da_imagem) # carregar a imagem na URL informada
            face_img = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # converte a imagem de BGR (padrão OCV) para tons de cinza
            face = face_cascade.detectMultiScale(face_img)
            # canto inferior esquerdo
            cie = face[0][0]
            # canto superior esquerdo
            cse = face[0][1]
            # canto inferior direito
            cid = cie + face[0][2]
            # canto superior direito
            csd = cse + face[0][3]
    
            roi_img = face_img[cse:csd, cie:cid]
            img = cv2.resize(roi_img,(150, 150), interpolation = cv2.INTER_CUBIC)
            return img
    
        imagem = detectar_roi(imagem_carregada)
        imagem = np.array(imagem)
        imagem = imagem.reshape(104, 150, 150, 1)
    
        # carregar o modelo
        model = load_model('BBi9FaceSentimentos.h5')
    
        # fazer a previsão
        y_pred = model.predict_classes(imagem)
    
        # imprimir a previsão
        return y_pred
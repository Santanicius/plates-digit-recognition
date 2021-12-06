import cv2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
import numpy as np
import glob
import math
from matplotlib import pyplot as plt



imagefiles = glob.glob("cars_plates/*")
""" imagefiles = ["cars_plates/26112002071048.jpg", "cars_plates/26112002071458.jpg", "cars_plates/26112002071812.jpg"] """
hist_list_dict = []
for filename in imagefiles:
    img = cv2.imread(filename)
    img_copia = img.copy()

    img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calcula o histograma da imagem cinza
    hist, bins = np.histogram(img_cinza.ravel(),256,[127,256])
    
    # Pega a média das arestas bins 
    mean_bin = (np.sum(hist * np.diff(bins))/len(hist))
    
    
    # Verifica a media das areastas bins para aplicar o CLAHE:
    # improving local contrast with => Equalização de histograma adaptativo limitado por contraste (Ajuda nas piores anomalias)
    # Aplicando uma grid de determinado tamanho para melhorar o contraste local assim ajudando a aplicar o treshold de OTSU
    if 168.00 < mean_bin < 170.00 or 58 < mean_bin < 60.00:
        GRID_SIZE = 10
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(GRID_SIZE,GRID_SIZE))
        img_cinza = clahe.apply(img_cinza)
    elif 86.17 < mean_bin < 86.19 or 139.00 < mean_bin < 140.00:
        GRID_SIZE = 40
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(GRID_SIZE,GRID_SIZE))
        img_cinza = clahe.apply(img_cinza)
    elif 86.60 < mean_bin < 86.80:
        GRID_SIZE = 30
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(GRID_SIZE,GRID_SIZE))
        img_cinza = clahe.apply(img_cinza)
    elif 86 < mean_bin < 88:
        GRID_SIZE = 15
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(GRID_SIZE,GRID_SIZE))
        img_cinza = clahe.apply(img_cinza)
    elif 37.00 < mean_bin < 38.00 or 161.10 < mean_bin < 161.20:
        GRID_SIZE = 30
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(GRID_SIZE,GRID_SIZE))
        img_cinza = clahe.apply(img_cinza)
    elif 161.00 < mean_bin < 162.00:
        GRID_SIZE = 20
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(GRID_SIZE,GRID_SIZE))
        img_cinza = clahe.apply(img_cinza)
    elif 69.00 < mean_bin < 70.00:
        GRID_SIZE = 12
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(GRID_SIZE,GRID_SIZE))
        img_cinza = clahe.apply(img_cinza)
    
    img_cinza = cv2.GaussianBlur(img_cinza, (3,3), 0.2)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img_laplacian = cv2.filter2D(img_cinza, -1, kernel) 
    img_gauss = cv2.GaussianBlur(img_laplacian, (3,3), 0)
    
    ret, thresh_otsu = cv2.threshold(img_gauss, 127, 255, type=cv2.THRESH_OTSU)
    
    contours, hierarchy = cv2.findContours(thresh_otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    
    #obtendo apenas os retangulos de certo tamanho
    listIniRect = []
    for rec in contours:
        x,y,w,h = cv2.boundingRect(rec)
        if h > 17 and h < 35 and w > 3 and w < 50:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 1)
            listIniRect.append([x,y,x+w,y+h])
            
    #eliminando os retangulos sobrepostos
    listOverRect = []
    for i in range(len(listIniRect)):
        flag = 0
        for j in range(len(listIniRect)):
            if i!=j:
                if (listIniRect[i][0]>listIniRect[j][0] and listIniRect[i][0]<listIniRect[j][2] and
                    (listIniRect[i][1]>=listIniRect[j][1] and listIniRect[i][1]<=listIniRect[j][3] or
                     listIniRect[i][3]>=listIniRect[j][1] and listIniRect[i][3]<=listIniRect[j][3])):
                    cv2.rectangle(img, (listIniRect[i][0], listIniRect[i][1]),(listIniRect[i][2], listIniRect[i][3]),(0,255,),1)                    
                    flag=1
        if flag==0:
            listOverRect.append(listIniRect[i])

    #eliminando os retangulos nao alinhados com a maioria
    listNotAlign = []
    for i in range(len(listOverRect)):
        cont=0
        for j in range(len(listOverRect)):
            if i != j and math.fabs(listOverRect[i][1] - listOverRect[j][1]) < 10 and math.fabs(listOverRect[i][3] - listOverRect[j][3]) < 10:
                cont += 1
        if cont>=len(listOverRect)*.4:
            listNotAlign.append(listOverRect[i])        

    for i in range(len(listNotAlign)):
        cv2.rectangle(img, (listNotAlign[i][0], listNotAlign[i][1]),(listNotAlign[i][2], listNotAlign[i][3]),(0,0,255),2) 
    
    # Verificar se tem alguem na lista dos retangulos nao alinhados com a maioria se nao usa a lista anterior
    if len(listNotAlign) > 0:
        xi = listNotAlign[0][0]
        yi = listNotAlign[0][1]
        xf = listNotAlign[0][2]
        yf = listNotAlign[0][3]

        for i in range(len(listNotAlign)):
            if listNotAlign[i][0] < xi:
                xi = listNotAlign[i][0]
            if listNotAlign[i][1] < yi:
                yi = listNotAlign[i][1]    
            if listNotAlign[i][2] > xf:
                xf = listNotAlign[i][2]
            if listNotAlign[i][3] > yf:
                yf = listNotAlign[i][3]
    else:
        xi = listOverRect[0][0]
        yi = listOverRect[0][1]
        xf = listOverRect[0][2]
        yf = listOverRect[0][3]

        for i in range(len(listOverRect)):
            if listOverRect[i][0] < xi:
                xi = listOverRect[i][0]
            if listOverRect[i][1] < yi:
                yi = listOverRect[i][1]    
            if listOverRect[i][2] > xf:
                xf = listOverRect[i][2]
            if listOverRect[i][3] > yf:
                yf = listOverRect[i][3]
        
    yi -= 7
    yf += 7
    dif = int((250 - (xf - xi)) / 2)
    xi=max(0, xi - dif)
    xf=min(img.shape[1] - 1,xf + dif)
    cv2.rectangle(img, (xi, yi), (xf, yf),(255, 255, 0), 2)

    img_placa = img_copia[yi:yf,xi:xf]
    
    """     
    img_placa_cinza = cv2.cvtColor(img_placa, cv2.COLOR_BGR2GRAY)

    ret_placa, thresh_placa = cv2.threshold(img_placa_cinza, 105, 255, type=cv2.THRESH_OTSU) """

    """ 
    cv2.imshow(filename + "_placa", img_placa)
    cv2.imshow(filename + "_carro", img) """
    
    """ cv2.imshow(filename + "_carro", img) """

    cv2.waitKey(0)




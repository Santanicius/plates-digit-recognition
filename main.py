import cv2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
import numpy as np
import glob
import math

imagefiles = glob.glob("estragadas/*")
for filename in imagefiles:
    img = cv2.imread(filename)
    img_copia = img.copy()


    img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # improving local contrast with => Equalização de histograma adaptativo limitado por contraste (Ajuda nas anomalias)
    GRID_SIZE = 40
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(GRID_SIZE,GRID_SIZE))
    img_cinza = clahe.apply(img_cinza)
    
    
    img_cinza = cv2.GaussianBlur(img_cinza, (3,3), 0.2)
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    img_laplacian = cv2.filter2D(img_cinza, -1, kernel) 
    img_gauss = cv2.GaussianBlur(img_laplacian, (3,3), 0)                                                                                                                                                                                                                                                                   
    """ 
        Da uma melhorada pra alguns casos
        ret, thresh = cv2.threshold(img_gauss, 130, 255, type=cv2.THRESH_TOZERO)
    """
    ret, thresh_otsu = cv2.threshold(img_gauss, 130, 255, type=cv2.THRESH_OTSU)
    blur = cv2.bilateralFilter(thresh_otsu,9,75,75)
    
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
                if listIniRect[i][0]>listIniRect[j][0] and listIniRect[i][0]<listIniRect[j][2] and (listIniRect[i][1]>=listIniRect[j][1] and listIniRect[i][1]<=listIniRect[j][3] or listIniRect[i][3]>=listIniRect[j][1] and listIniRect[i][3]<=listIniRect[j][3]):
                    cv2.rectangle(img, (listIniRect[i][0], listIniRect[i][1]),(listIniRect[i][2], listIniRect[i][3]),(0,255,255),1)                    
                    flag=1
        if flag==0:
            listOverRect.append(listIniRect[i])
    cv2.imshow(filename,img)
    cv2.waitKey(0)
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
        cv2.rectangle(img, (listNotAlign[i][0], listNotAlign[i][1]),(listNotAlign[i][2], listNotAlign[i][3]),(0,0,255),3) 

    cv2.imshow(filename,img)
    cv2.waitKey(0)
    # Verificar se tem mais retangulo pra direita ou esquerda...
    if len(listNotAlign) < 7:
        #obtendo a regiao da placa
        xi = listNotAlign[0][0]-10
        yi = listNotAlign[0][1]-10
        xf = listNotAlign[0][2]+10
        yf = listNotAlign[0][3]+10
    else:
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

    yi -= 10
    yf += 10
    dif = int((250 - (xf - xi)) / 2)
    xi=max(0, xi - dif)
    xf=min(img.shape[1] - 1,xf + dif)
    cv2.rectangle(img, (xi, yi), (xf, yf),(255, 255, 0), 2)

    img_placa=img_copia[yi:yf,xi:xf]
    if len(listNotAlign) < 7:
        nameimg = "estragadas\\"+filename.split("\\")[1]
        print(nameimg, "\n")
        cv2.imwrite(nameimg, img_copia)
    cv2.imshow(filename + "_placa", img_placa)

    cv2.waitKey(0)




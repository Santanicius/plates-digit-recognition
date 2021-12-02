import cv2
import numpy as np
import glob
import math

imagefiles = glob.glob("estragadas/*")
for filename in imagefiles:
    img = cv2.imread(filename)
    img_copia = img.copy()


    img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_cinza = cv2.GaussianBlur(img_cinza, (3,3), 0.2)
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    img_laplacian = cv2.filter2D(img_cinza, -1, kernel) 

    img_gauss = cv2.GaussianBlur(img_laplacian, (3,3), 0)

    ret, thresh = cv2.threshold(img_gauss, 120, 255, type=cv2.THRESH_OTSU)
    

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

    #obtendo apenas os retangulos de certo tamanho
    lista = []
    for rec in contours:
        x,y,w,h = cv2.boundingRect(rec)
        if h>17 and h<35 and w>3 and w<50:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 1)
            lista.append([x,y,x+w,y+h])

    #eliminando os retangulos sobrepostos
    lista_2 = []
    for i in range(len(lista)):
        flag = 0
        for j in range(len(lista)):
            if i!=j:
                if lista[i][0]>lista[j][0] and lista[i][0]<lista[j][2] and (lista[i][1]>=lista[j][1] and lista[i][1]<=lista[j][3] or lista[i][3]>=lista[j][1] and lista[i][3]<=lista[j][3]):
                    cv2.rectangle(img, (lista[i][0], lista[i][1]),(lista[i][2], lista[i][3]),(0,255,255),1)                    
                    flag=1
        if flag==0:
            lista_2.append(lista[i])

    #eliminando os retangulos nao alinhados com a maioria
    lista_3 = []
    for i in range(len(lista_2)):
        cont=0
        for j in range(len(lista_2)):
            if i!=j and math.fabs(lista_2[i][1]-lista_2[j][1])<10 and math.fabs(lista_2[i][3]-lista_2[j][3])<10:
                cont+=1
        if cont>=len(lista_2)*.4:
            lista_3.append(lista_2[i])        


    for i in range(len(lista_3)):
        cv2.rectangle(img, (lista_3[i][0], lista_3[i][1]),(lista_3[i][2], lista_3[i][3]),(0,0,255),3) 

    #obtendo a regiao da placa
    xi=lista_3[0][0]
    yi=lista_3[0][1]
    xf=lista_3[0][2]
    yf=lista_3[0][3]

    for i in range(len(lista_3)):
        if lista_3[i][0]<xi:
            xi=lista_3[i][0]
        if lista_3[i][1]<yi:
            yi=lista_3[i][1]    
        if lista_3[i][2]>xf:
            xf=lista_3[i][2]
        if lista_3[i][3]>yf:
            yf=lista_3[i][3]

    yi-=3
    yf+=3
    dif=int((250-(xf-xi))/2)
    xi=max(0,xi-dif)
    xf=min(img.shape[1]-1,xf+dif)
    cv2.rectangle(img, (xi,yi),(xf,yf),(0,0,0),3)

    img_placa=img_copia[yi:yf,xi:xf]
    if len(lista_3) < 7:
        nameimg = "estragadas\\"+filename.split("\\")[1]
        print(nameimg,"\n")
        cv2.imwrite(nameimg, img_copia)
    cv2.imshow(filename+"_placa",img_placa)

    cv2.imshow(filename,img)
    cv2.waitKey(0)




import cv2

NUMEROS = 1
LETRAS = 2

class NaoOcorre:
    def __init__(self):
        self._00 = 1
        self._01 = 1
        self._10 = 1
        self._11 = 1

class Classe:
    def __init__(self, caractere, n_dim):
        self.caractere = caractere
        self.n_dim = n_dim
        self.n_restricoes = 0
        self.NOC = []
        for i in range(n_dim - 1):
            self.NOC.append(NaoOcorre())
        
class ClassificacaoCaractere:
    def __init__(self,altura,largura,tipo,flag):
        self.altura = altura
        self.largura = largura
        self.tipo = tipo
        self.n_classes = 0
        self.n_dim = altura*largura
        self.classes = []

        if tipo == 1:
            dig_numeros = "0123456789"
            self.n_classes = 10
            for i in range(self.n_classes):
                self.classes.append(Classe(dig_numeros[i],self.n_dim))
            if flag=='S':
                self.monta_arq_aprendizado(dig_numeros,"numeros.png","numeros.txt")
            self.incializa_classificador_2pixels("numeros.txt")
        else:
            dig_letras = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            self.n_classes = 26
            for i in range(self.n_classes):
                self.classes.append(Classe(dig_letras[i],self.n_dim))
            if flag=='S':
                self.monta_arq_aprendizado(dig_letras,"letras.png","letras.txt")
            self.incializa_classificador_2pixels("letras.txt")

    def monta_arq_aprendizado(self,digitos,nome_img,nome_arq):
        img = cv2.imread(nome_img)
        arq = open(nome_arq,"a")

        img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_cinza, 127, 255, type=cv2.THRESH_BINARY)

        contours, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        lista = []
        for rec in contours:
            x,y,w,h = cv2.boundingRect(rec)
            if h>30 and h<40:
                #cv2.rectangle(img_cinza, (x, y), (x+w-1, y+h-1), (100,100,100), 1)
                lista.append([x, y, x+w-1, y+h-1])
       
        lista.sort()
        for i in range(len(lista)):
            xi = lista[i][0]
            yi = lista[i][1]
            xf = lista[i][2]
            yf = lista[i][3]
            img_dig = img_cinza[yi:yf,xi:xf]

            #cv2.imshow(''+str(i),img_dig)
            #cv2.waitKey(0)
            transicao = self.retornaTransicaoHorizontal(img_dig)
            arq.write(digitos[i]+'|'+transicao+"|\n")
        arq.close()

    def retornaTransicaoHorizontal(self,img):
        dim = (self.largura, self.altura)
        img_res = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        transicao = ""
        flag = True
        for i in range(self.altura):
            if flag:
                for j in range(self.largura):
                    if img_res[i][j] == 0:
                        transicao += '0'
                    else:
                        transicao += '1'
            else:
                for j in range(self.largura-1,-1,-1):
                    if img_res[i][j] == 0:
                        transicao += '0'
                    else:
                        transicao += '1'    
            flag = not flag    
        return transicao

    def incializa_classificador_2pixels(self,nome_arq):
        arq = open( nome_arq,'r')
        todos_dados = arq.readlines()

        pos = 0
        for linha in todos_dados:
            if pos < len(self.classes):
                self.classes[pos].n_restricoes = (self.n_dim - 1)*4
                lista = linha.split('|')
                transicao = lista[1]
                for j in range(self.n_dim -1):
                    if transicao[j]=='0' and transicao[j+1]=='0' and self.classes[pos].NOC[j]._00 == 1:
                        self.classes[pos].NOC[j]._00 = 0 #0=ocorre, 1=nao ocorre
                        self.classes[pos].n_restricoes-=1;
                    if transicao[j]=='0' and transicao[j+1]=='1' and self.classes[pos].NOC[j]._01 == 1:
                        self.classes[pos].NOC[j]._01 = 0 #0=ocorre, 1=nao ocorre
                        self.classes[pos].n_restricoes-=1;
                    if transicao[j]=='1' and transicao[j+1]=='0' and self.classes[pos].NOC[j]._10 == 1:
                        self.classes[pos].NOC[j]._10 = 0 #0=ocorre, 1=nao ocorre
                        self.classes[pos].n_restricoes-=1;
                    if transicao[j]=='1' and transicao[j+1]=='1' and self.classes[pos].NOC[j]._11 == 1:
                        self.classes[pos].NOC[j]._11 = 0 #0=ocorre, 1=nao ocorre
                        self.classes[pos].n_restricoes-=1;                    
                pos+=1
        arq.close()

    def reconheceCaractereTransicao_2pixels(self, transicao):
        cont_NOC = []
        caractere = ' '

        for i in range(self.n_classes):
            cont_NOC.append(0)
            for j in range(self.n_dim - 1):
                #se tem alguma restricao
                if self.classes[i].NOC[j]._00 == 1 or self.classes[i].NOC[j]._01 == 1 or self.classes[i].NOC[j]._10 == 1 or self.classes[i].NOC[j]._11 == 1:
                    if transicao[j]=='0' and transicao[j+1]=='0' and self.classes[i].NOC[j]._00 == 1:
                        cont_NOC[i]+=1
                    if transicao[j]=='0' and transicao[j+1]=='1' and self.classes[i].NOC[j]._01 == 1:
                        cont_NOC[i]+=1  
                    if transicao[j]=='1' and transicao[j+1]=='0' and self.classes[i].NOC[j]._10 == 1:
                        cont_NOC[i]+=1    
                    if transicao[j]=='1' and transicao[j+1]=='1' and self.classes[i].NOC[j]._11 == 1:
                        cont_NOC[i]+=1

        menor = self.n_dim
        pos = 0
        for i in range(self.n_classes):
            if cont_NOC[i] < menor:
                menor = cont_NOC[i]
                pos = i
        #se empatar, verificar a classe com mais restricoes
        maior = 0
        for i in range(self.n_classes):
            if menor == cont_NOC[i]:
                if self.classes[i].n_restricoes > maior:
                    maior = self.classes[i].n_restricoes
                    pos = i
        if menor < self.n_dim:
            caractere = self.classes[pos].caractere
        return caractere



'''
    #teste de reconhecimento de numeros
    img = cv2.imread("numeros.png") #1
    #img = cv2.imread("letras.png") #1

    img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_cinza, 127, 255, type=cv2.THRESH_BINARY)

    contours, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    lista = []
    for rec in contours:
        x,y,w,h = cv2.boundingRect(rec)
        if h>30 and h<40:
            #cv2.rectangle(img_cinza, (x, y), (x+w-1, y+h-1), (100,100,100), 1)
            lista.append([x, y, x+w-1, y+h-1])
       
    lista.sort()
    for i in range(len(lista)):
        xi = lista[i][0]
        yi = lista[i][1]
        xf = lista[i][2]
        yf = lista[i][3]
        img_dig = img_cinza[yi:yf,xi:xf]
        cv2.imshow(''+str(i),img_dig)
        transicao = cl.retornaTransicaoHorizontal(img_dig)
        print(cl.reconheceCaractereTransicao_2pixels(transicao))
        cv2.waitKey(0)        
'''

if __name__ == "__main__":
    cl = ClassificacaoCaractere(30,40,2,'S')
    img = cv2.imread("letra_l.png")
    

    img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_cinza, 127, 255, type=cv2.THRESH_BINARY)

    transicao = cl.retornaTransicaoHorizontal(thresh)
    print("Previsto: ",cl.reconheceCaractereTransicao_2pixels(transicao))

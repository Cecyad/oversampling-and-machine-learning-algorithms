import numpy as np 
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric
from operator import itemgetter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import math
from sklearn.ensemble import RandomForestClassifier
import sys

def contar(caden):
    medida=len(caden)
    nume=0
    contarr=0
    cont=0
    while nume < medida:
        contarr=len(caden[nume])
        cont=cont+contarr
        nume+=1
    return cont

def knn(X_train,y_train,X_test,n):
    algoritmo = KNeighborsClassifier(n_neighbors = n,p=1)
    algoritmo.fit(X_train,y_train.values.ravel())
    c,e=algoritmo.kneighbors(X_test,n_neighbors = n, return_distance=True)
    y_pred = algoritmo.predict(X_test)
    return e,y_pred


def orden(cadenaa):
    medida=len(cadenaa)
    nume=0
    bandera=0
    while nume < medida:
        cadenaa[nume]=sorted(cadenaa[nume])
        nume+=1
    cadenaa=sorted(cadenaa)
    return cadenaa

def buscar(r,pruebaex):
    ln=len(pruebaex)
    i=0
    z=1
    while i<ln:
        if r==pruebaex[i]:
            i=ln
            z=0
        i+=1
    return z
def vacios(pruebaex):
    i=0
    while i< len(pruebaex):
        t=len(pruebaex[i])
        if t==0:
            pruebaex.pop(i)
            i-=1
        i+=1
    return pruebaex

def crearsub(resulta,suma,porcen,subgrupoprueba,raro,disyuncioness,fronterass,seguross):
    i=0
    rescon=len(resulta)
    por=0
    #contador=0
    contarprue=0
    pruebaexr=[]
    pruebaexd=[]
    pruebaexs=[]
    pruebaexf=[]

    while i < rescon:
        p=0
        z=0
        if i == rescon-1:
            por=porcen-contarprue
            #print("numero es-------------------", por)
            #print(pruebaex)
            if resulta[i][0]=="r":
                
                while p < por:
                    z=0
                    while z<1:
                        r=random.randint(0,(len(raro)-1))
                        z=buscar(raro[r],pruebaexr)
                    subgrupoprueba.append(raro[r])
                    pruebaexr.append(raro[r])
                    p+=1
            if resulta[i][0]=="d":
                num=0
                while p < por:
                    l=len(disyuncioness[num])
                    aux=disyuncioness[num]
                    y=0
                    
                    #disyuncioness.pop(num)
                    while y < l:
                        z=0
                        while z<1:
                            r=random.randint(0,(len(aux)-1))
                            z=0
                            z=buscar(aux[r],pruebaexd)
                            if z==0:
                                l-=1
                            if l==0:
                                break
                        if z==1:
                            subgrupoprueba.append(aux[r])
                            pruebaexd.append(aux[r])
                            p+=1
                        y+=1
                        if p==por:
                            break
                    num+=1
            if resulta[i][0]=="s":
                num=0
                u=0
                while p < por:
                    l=len(seguross[num])
                    aux=seguross[num]
                    y=0
                    
                    #seguross.pop(num)
                    while y < l:
                        z=0
                        while z<1:
                            r=random.randint(0,(len(aux)-1))
                            z=0
                            z=buscar(r,pruebaexs)
                            if z==0:
                                l-=1
                            if l==0:
                                break
                        if z==1:
                            subgrupoprueba.append(aux[r])
                            pruebaexs.append(aux[r])
                            p+=1
                        y+=1
                        if p==por:
                            break
                    num+=1
            if resulta[i][0]=="f":
                num=0
                while p < por:
                    l=len(fronterass[num])
                    aux=fronterass[num]
                    y=0
                    #fronterass.pop(num)
                    while y < l:
                        z=0
                        while z<1:
                            r=random.randint(0,(len(aux)-1))
                            z=0
                            z=buscar(aux[r],pruebaexf)
                            if z==0:
                                l-=1
                            if l==0:
                                break
                        if z==1:
                            subgrupoprueba.append(aux[r])
                            pruebaexf.append(aux[r])
                            p+=1
                        y+=1
                        if p==por:
                            break
                    num+=1
        else:
            port=(resulta[i][1]*100)/suma
            port=math.ceil(port)
            por=(port*porcen)/100
            #poren=((port*entrenamin)/100)+0.5
            por=math.ceil(por)
            contarprue=contarprue+por
            if resulta[i][0]=="r":
                while p < por:
                    z=0
                    while z<1:
                        r=random.randint(0,(len(raro)-1))
                        z=0
                        z=buscar(raro[r],pruebaexr)
                    subgrupoprueba.append(raro[r])
                    pruebaexr.append(raro[r])
                    #raro.pop(p)
                    p+=1
            if resulta[i][0]=="d":
                num=0
                while p < por:
                    l=len(disyuncioness[num])
                    aux=disyuncioness[num]
                    y=0
                    #disyuncioness.pop(num)
                                            
                    while y < l:
                        z=0
                        while z<1:
                            r=random.randint(0,(len(aux)-1))
                            #print("en d, r es:",r)
                            z=0
                            z=buscar(aux[r],pruebaexd)
                            if z==0:
                                l-=1
                            if l==0:
                                break
                        if z==1:
                            subgrupoprueba.append(aux[r])
                            pruebaexd.append(aux[r])
                            p+=1
                        y+=1
                        if p==por:
                            break
                    num+=1
            if resulta[i][0]=="s":
                num=0
                while p < por:
                    l=len(seguross[num])
                    aux=seguross[num]
                    y=0 
                    #seguross.pop(num)
                    #seguross.pop()
                    while y < l:
                        z=0
                        while z<1:
                            r=random.randint(0,(len(aux)-1))
                            z=0
                            z=buscar(aux[r],pruebaexs)
                            if z==0:
                                l-=1
                            if l==0:
                                break
                        if z==1:
                            subgrupoprueba.append(aux[r])
                            pruebaexs.append(aux[r])
                            p+=1
                        y+=1
                        if p==por:
                            break
                    num+=1
            if resulta[i][0]=="f":
                num=0
                while p < por:
                    l=len(fronterass[num])
                    aux=fronterass[num]
                    y=0 
                    #fronterass.pop(num)
                    while y < l:
                        z=0
                        while z<1:
                            r=random.randint(0,(len(aux)-1))
                            #print("en frontera r es: ", r)
                            z=0
                            z=buscar(aux[r],pruebaexf)
                            if z==0:
                                l-=1
                            if l==0:
                                break
                        if z==1:
                            subgrupoprueba.append(aux[r])
                            pruebaexf.append(aux[r])
                            p+=1
                        y+=1
                        if p==por:
                            break
                    num+=1
        i+=1

    return subgrupoprueba,pruebaexs,pruebaexr,pruebaexd,pruebaexf

def crearentrenammienti(resulta,suma,subgrupoentrenamiento,entrenamin,raro,disyuncioness,seguross,fronterass):
    i=0
    rescon=len(resulta)
    por=0
    #contador=0
    contarprue=0

    while i < rescon:
        p=0
        z=0
        if i == rescon-1:
            por=entrenamin-contarprue
            if resulta[i][0]=="r":
                
                while p < por:
                    z=0
                    subgrupoentrenamiento.append(raro[p])
                    p+=1

            if resulta[i][0]=="d":
                num=0
                while p < por:
                    l=len(disyuncioness[num])
                    aux=disyuncioness[num]
                    y=0
                    
                    #disyuncioness.pop(num)
                    while y < l:
                        z=0
                        subgrupoentrenamiento.append(aux[y])
                        p+=1
                        y+=1
                        if p==por:
                            break
                    num+=1
            if resulta[i][0]=="s":
                num=0
                u=0
                while p < por:
                    l=len(seguross[num])
                    aux=seguross[num]
                    y=0
                    
                    while y < l:
                        z=0
                        subgrupoentrenamiento.append(aux[y])
                        p+=1
                        y+=1
                        if p==por:
                            break
                    num+=1

            if resulta[i][0]=="f":
                num=0
                while p < por:
                    l=len(fronterass[num])
                    aux=fronterass[num]
                    y=0
                    while y < l:
                        z=0
                        subgrupoentrenamiento.append(aux[y])
                        p+=1
                        y+=1
                        if p==por:
                            break
                    num+=1
        else:
            port=(resulta[i][1]*100)/suma
            port=math.ceil(port)
            por=(port*entrenamin)/100
            por=math.ceil(por)
            contarprue=contarprue+por
            if resulta[i][0]=="r":
                while p < por:
                    z=0
                    r=random.randint(0,(len(raro)-1))
                    subgrupoentrenamiento.append(raro[r])
                    p+=1
            if resulta[i][0]=="d":
                num=0
                while p < por:
                    l=len(disyuncioness[num])
                    aux=disyuncioness[num]
                    y=0
                                            
                    while y < l:
                        z=0
                        subgrupoentrenamiento.append(aux[y])
                        p+=1
                        y+=1
                        if p==por:
                            break
                    num+=1
            if resulta[i][0]=="s":
                num=0
                while p < por:
                    l=len(seguross[num])
                    aux=seguross[num]
                    y=0 
                    while y < l:
                        z=0
                        subgrupoentrenamiento.append(aux[y])
                        p+=1
                        y+=1
                        if p==por:
                            break
                    num+=1
            if resulta[i][0]=="f":
                num=0
                while p < por:
                    l=len(fronterass[num])
                    aux=fronterass[num]
                    y=0 
                    while y < l:
                        z=0
                        subgrupoentrenamiento.append(aux[y])
                        p+=1
                        y+=1
                        if p==por:
                            break
                    num+=1
        i+=1

    return subgrupoentrenamiento

def generaro(cadena,df,ll):
    listaaa=[]
    contarr=0
    r=0
    vol1,u=df.shape
    s=0
    dr=[]
    while contarr < ll:
        z=cadena[contarr]
        pi=0
        while pi < u-1:
            r=df.loc[z][pi]+0.000001
            listaaa.append(r)
            pi+=1
        listaaa.append(1)
        #print("la lista es:::::.:::::::::.",listaaa)
        s=len(df)
        dr.append(listaaa)
        df.loc[s]=listaaa
        cadena.append(s)
        listaaa=[]
        contarr+=1
    return cadena,dr

def generarinstt(cadena,df,ll):
    dissss=[]
    listaaa=[]
    test=[]
    testy=[]
    dr=[]
    i=0
    sol=0
    m1=0
    contarr=0
    vol1,u=df.shape
    ii=0
    while contarr < ll:
        while i< len(cadena):
            #print("es=========",cadena[i])
            t=len(cadena[i])
            #print("t======",t)
            p=cadena[i]
            #print(p)
            y=0 
            r=random.randint(0,(t-1))
            while y<t:
                if y == r:
                    y+=1
                    if y==t:
                       break
                z=p[y]
                #print(z)
                pi=0
                while pi < u-1:
                    listaaa.append(df.loc[z][pi])
                    pi+=1
                #print("laaaaaaaaaaaaaaaaa listaaaaaaaaaaaaaaaaaaaa es:",listaaa)
                y+=1
                test.append(listaaa)
                testy.append(df.loc[z][pi])
                listaaa=[]
            lon=len(test)
            pi=0
            # while pi < lon:
            #print("mm=======",len(test))
            while pi < u-1:
                mm=len(test)
                rr=0
                m1=0
                while rr<mm:
                    m1=m1+test[rr][pi]
                    rr+=1
                #print("suma = ",m1)
                m1=m1/mm
                m1=(math.ceil(m1))
                #print("res =", m1)
                listaaa.append(m1)
                pi+=1
            listaaa.append(1)
            dr.append(listaaa)
            sol=len(df)
            df.loc[sol]=listaaa
            #print(cadena)
            cadena[i].append(sol)
            #print(cadena)
            dissss.append(listaaa)
            numeroo=len(df)
            contarr+=1
            if contarr==ll:
                break
            listaaa=[]
            test=[]
            i+=1
        i=0
    return cadena,dr

def generarinst(cadena,df,ll):
    dissss=[]
    listaaa=[]
    test=[]
    testy=[]
    i=0
    sol=0
    dr=[]
    m1=0
    contarr=0
    vol1,u=df.shape
    ii=0
    while contarr < ll:
        while i< len(cadena):
            #t=len(cadena[i])
            p=cadena[i]
            y=0 
            r=0
            while y<2:
                z=p[r]
                pi=0
                while pi < u-1:
                    listaaa.append(df.loc[z][pi])
                    pi+=1
                #print("laaaaaaaaaaaaaaaaa listaaaaaaaaaaaaaaaaaaaa es:",listaaa)
                y+=1
                test.append(listaaa)
                testy.append(df.loc[z][pi])
                listaaa=[]
                r=random.randint(0,len(cadena[i])-1)
            lon=len(test)
            pi=0
            # while pi < lon:
            while pi < u-1:
                mm=len(test)
                rr=0
                m1=0
                while rr<mm:
                    m1=m1+test[rr][pi]
                    rr+=1
                m1=m1/2
                m1=math.floor(m1)
                listaaa.append(m1)
                pi+=1
            listaaa.append(1)
            dr.append(listaaa)
            sol=len(df)
            df.loc[sol]=listaaa
            cadena[i].append(sol)
            dissss.append(listaaa)
            numeroo=len(df)
            contarr+=1
            if contarr==ll:
                break
            listaaa=[]
            test=[]
            i+=1
        i=0
    return cadena,dr

#valor=1

a = int(sys.argv[1])

df=pd.read_csv('completo.csv',header=None)
subgrupoprueba=pd.read_csv('test.csv',header=None)

#nuevo representara el subgrupo de entrenamiento
nuevoo=pd.concat([df,subgrupoprueba], axis=0)
nuevoo.reset_index(drop=True, inplace=True)
nuevo=nuevoo.drop_duplicates( keep=False)
nuevo.reset_index(drop=True, inplace=True)

#ANALISIS DEL CONJUNTO DE DATOS.
xp,xpp=df.shape
xpp=xpp-1

df2=df[df.iloc[:, -1]==1]
xd,pddd=df2.shape

x_train=df[[0,1,2]]
y_train=df[[3]]
x_test=df2[[0,1,2]]
y_test = df2[[3]]


sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)


e,y_pred=knn(X_train,y_train,X_test,10)
#print(e)


z=np.zeros((xd, 10))
i=0
t=0
p=0
k=10
conj = {}
seguro=[]
seguros=[]
frontera=[]
fronteras=[]
disyuncion=[]
disyunciones=[]
raro=[]
conjunto=[]
co=0
dis=0

while i < xd:
    
    z[i,0]=y_train.loc[e[i,0]]
    z[i,1]=y_train.loc[e[i,1]]
    z[i,2]=y_train.loc[e[i,2]]
    z[i,3]=y_train.loc[e[i,3]]
    z[i,4]=y_train.loc[e[i,4]]
    z[i,5]=y_train.loc[e[i,5]]
    z[i,6]=y_train.loc[e[i,6]]
    z[i,7]=y_train.loc[e[i,7]]
    z[i,8]=y_train.loc[e[i,8]]
    z[i,9]=y_train.loc[e[i,9]]
    p=z[i,0]+z[i,1]+z[i,2]+z[i,3]+z[i,4]+z[i,5]+z[i,6]+z[i,7]+z[i,8]+z[i,9]
    if p<(k/2):
        if p<=1:
            raro.append(e[i,0])
        else:
            bu=np.where(z[i]>0)[0]
            v=0
            
            while v < p:
                disyuncion.append(e[i,bu][v])
                v+=1
            disyunciones.append(disyuncion)
            disyuncion=[]
    else:
        if p<=6:
            bu=np.where(z[i]>0)[0]
            v=0
            while v < p:
                frontera.append(e[i,bu][v])
                v+=1
            fronteras.append(frontera)
            frontera=[]
        else:
            bu=np.where(z[i]>0)[0]
            v=0
            while v < p:
                seguro.append(e[i,bu][v])
                v+=1
            seguros.append(seguro)
            seguro=[]
                
    
    p=0
    t+=1
    i+=1



disyuncioness=orden(disyunciones)

dis=[]
i=0

while i< len(disyuncioness):
    t=len(disyuncioness[i])
    y=0
    x=0
    while y<t:
        z=buscar(disyuncioness[i][x],dis)
        if z==1:
            dis.append(disyuncioness[i][x])
        else:
            disyuncioness[i].pop(x)
            x-=1
        x+=1
        y+=1
    i+=1

disyuncioness=vacios(disyuncioness)


i=0

while i<len(disyuncioness):
    t=len(disyuncioness[i])
    if t==1:
        disyuncioness[i-1].append(disyuncioness[i][0])
        disyuncioness.pop(i)
        i-=1
    i+=1



seguross=orden(seguros)

dis=[]
i=0

while i< len(seguross):
    t=len(seguross[i])
    y=0
    x=0
    while y<t:
        z=buscar(seguross[i][x],dis)
        if z==1:
            dis.append(seguross[i][x])
        else:
            seguross[i].pop(x)
            x-=1
        x+=1
        y+=1
    i+=1

seguross=vacios(seguross)

i=0

while i<len(seguross):
    t=len(seguross[i])
    if t==1:
        seguross[i-1].append(seguross[i][0])
        seguross.pop(i)
        i-=1
    i+=1


fronterass=orden(fronteras)

dis=[]
i=0

while i< len(fronterass):
    t=len(fronterass[i])
    y=0
    x=0
    while y<t:
        z=buscar(fronterass[i][x],dis)
        if z==1:
            dis.append(fronterass[i][x])
        else:
            fronterass[i].pop(x)
            x-=1
        x+=1
        y+=1
    i+=1

fronterass=vacios(fronterass)



i=0

while i<len(fronterass):
    t=len(fronterass[i])
    if t==1:
        fronterass[i-1].append(fronterass[i][0])
        fronterass.pop(i)
        i-=1
    i+=1



crear=(len(df[df.iloc[:, -1]==0]))-(len(df[df.iloc[:, -1]==1]))



resulta={}
resulta["r"]=len(raro)
resulta["d"]=contar(disyuncioness)
resulta["s"]=contar(seguross)
resulta["f"]=contar(fronterass)
resulta=sorted(resulta.items(), key=lambda item: item[1])
suma=resulta[0][1]+resulta[1][1]+resulta[2][1]+resulta[3][1]



#GENERAR INSTANCIAS

i=0
r=1
n=1
e=len(disyuncioness)
res={}
rescon=len(resulta)
por=0
contador=0
contarprue=0
i=0

while i < rescon:
    p=0
    if i == rescon-1:
        por=crear-contarprue
        if resulta[i][0]=="r":
            res["r"]=resulta[i][1]
            contarprue=contarprue+resulta[i][1]
        if resulta[i][0]=="d":
            res["d"]=por
        if resulta[i][0]=="s":
            res["s"]=por
        if resulta[i][0]=="f":
            res["f"]=por
    else:
        port=(resulta[i][1]*100)/suma
        port=math.ceil(port)
        por=(port*crear)/100
        por=math.ceil(por)
        if resulta[i][0]=="r":
            res["r"]=resulta[i][1]
            contarprue=contarprue+resulta[i][1]
        if resulta[i][0]=="d":
            res["d"]=por
            contarprue=contarprue+por
        if resulta[i][0]=="s":
            res["s"]=por
            contarprue=contarprue+por
        if resulta[i][0]=="f":
            res["f"]=por
            contarprue=contarprue+por
    i+=1


ii=res.get("r")+res.get("s")+res.get("d")+res.get("f")
#print("suma=",ii)

raro,rr=generaro(raro,df,res.get("r"))
rr=pd.DataFrame(rr)

if a==1:
    seguross,di=generarinstt(seguross,df,res.get("s"))
if a==2:
    seguross,di=generarinst(seguross,df,res.get("s"))

di=pd.DataFrame(di)
#print(seguross)
#print(di)

dr=pd.concat([di,rr], axis=0)

if a==1:
    disyuncioness,dd=generarinstt(disyuncioness,df,res.get("d"))
if a==2:
    disyuncioness,dd=generarinst(disyuncioness,df,res.get("d"))

dd=pd.DataFrame(dd)

inst=pd.concat([dr,dd], axis=0)

if a==1:
    fronterass,ff=generarinstt(fronterass,df,res.get("f"))

if a==2:
    fronterass,ff=generarinst(fronterass,df,res.get("f"))


ff=pd.DataFrame(ff)
insta=pd.concat([inst,ff], axis=0)

insta.reset_index(drop=True, inplace=True)


trainn=pd.concat([insta,nuevo], axis=0)
trainn.reset_index(drop=True, inplace=True)


train_x=pd.DataFrame(trainn[[0,1,2]])
train_y=pd.DataFrame(trainn[[3]])

test_x=pd.DataFrame(subgrupoprueba[[0,1,2]])
test_y=pd.DataFrame(subgrupoprueba[[3]])


sc=StandardScaler()
X_train=sc.fit_transform(train_x)
X_test=sc.transform(test_x)


e,y_pred=knn(X_train,train_y,X_test,5)


medida=precision_recall_fscore_support(test_y, y_pred, average=None)


accuracyy=accuracy_score(test_y, y_pred)



mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=10000)
mlp.fit(X_train, train_y.values.ravel())

predictions = mlp.predict(X_test)

medida=precision_recall_fscore_support(test_y, predictions, average=None)
#print(medida)
accuracyy=accuracy_score(test_y, predictions)





tree = DecisionTreeClassifier()
tree.fit(X_train, train_y.values.ravel())
pre=tree.predict(X_test)
#print()
#print("Árboles de decisión")
medida=precision_recall_fscore_support(test_y, pre, average=None)
#print("medida F")
#print(medida)
accuracyy=accuracy_score(test_y, pre)
#print("Accuracy")
#print(accuracyy)
#print()


#print()
#print("Bosque aleatorio")
bosque=RandomForestClassifier()
bosque.fit(X_train, train_y.values.ravel())
predecir=bosque.predict(X_test)
medida=precision_recall_fscore_support(test_y, predecir, average=None)

accuracyy=accuracy_score(test_y, predecir)




modelo = SVC(C = 100, kernel = 'linear', random_state=123)
modelo.fit(X_train, train_y.values.ravel())
predicciones = modelo.predict(X_test)

medida=precision_recall_fscore_support(test_y, predicciones, average=None)


accuracyy=accuracy_score(test_y, predicciones)






voto=[]
voto.append(y_pred)
voto.append(predictions)
voto.append(pre)
voto.append(predecir)
voto.append(predicciones)

voto1 = np.array(voto)
#t2 = t2.T

lu=len(voto1[0])
i=0
voto=[]
punto=0
while i < lu:
    punto=voto1[0][i]+voto1[1][i]+voto1[2][i]+voto1[3][i]+voto1[3][i]
    if punto>2:
        voto.append(1)
    else:
        voto.append(0)
    i+=1
    punto=0



resulados=precision_recall_fscore_support(test_y, voto, average=None)

accuracyy=accuracy_score(test_y, voto)
print(resulados[0][0],resulados[0][1],resulados[1][0],resulados[1][1],resulados[2][0],resulados[2][1],accuracyy, sep=",")




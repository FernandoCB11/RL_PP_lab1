import random
import numpy
import pandas as pd
import gc
from datetime import datetime

class MDPClass:
    def __init__ (self, nX, nY, new_A, ambiente):
        self.S  = numpy.array(range(nX*nY))
        self.nX = nX
        self.nY = nY
        self.A  = new_A    #Ações
        self.Ambiente  = ambiente
        self.pol = None    #Politica

        TransicoesL = pd.read_fwf("../"+ambiente+"/Action_Leste.txt", colspecs="infer", header=None)
        TransicoesL.columns = ["Origem","Destino","Probabilidade"]
        TransicoesN = pd.read_fwf("../"+ambiente+"/Action_Norte.txt", colspecs="infer", header=None)
        TransicoesN.columns = ["Origem","Destino","Probabilidade"]
        TransicoesO = pd.read_fwf("../"+ambiente+"/Action_Oeste.txt", colspecs="infer", header=None)
        TransicoesO.columns = ["Origem","Destino","Probabilidade"]
        TransicoesS = pd.read_fwf("../"+ambiente+"/Action_Sul.txt", colspecs="infer", header=None)
        TransicoesS.columns = ["Origem","Destino","Probabilidade"]

        self.Custo = pd.read_fwf("../"+ambiente+"/Cost.txt", colspecs="infer", header=None).to_numpy()
        ##self.Custo.columns = ["Custo"]

        self.Transicoes = {"N": TransicoesN, "S" : TransicoesS, "L" : TransicoesL, "O" : TransicoesO}

        self.acessosTransicoes = 0

        self.matrizAlcancavel = numpy.zeros((len(self.S),len(self.S)))

    def zeraAcessos(self):
        self.acessosTransicoes = 0

    def acessosExecutados(self):
        return  self.acessosTransicoes

    def goalStates(self):
        busca = numpy.where(self.Custo == 0)[0]
        busca = busca + 1
        return busca.tolist()

    def transicao(self, s, a):
        self.acessosTransicoes = self.acessosTransicoes + 1
        proximos_index = self.Transicoes[self.A[a]]['Origem'] == (s + 1)
        proximos = self.Transicoes[self.A[a]][proximos_index==True]

        retorno_s = numpy.zeros(proximos.shape[0],dtype=int)
        retorno_p = numpy.zeros(proximos.shape[0])
        i = 0
        for index, row in proximos.iterrows():
            retorno_s[i] = int(row['Destino'])-1
            retorno_p[i] = row['Probabilidade']
            i = i+1

        return retorno_s, retorno_p

    def custo(self, s):
        return float(self.Custo[s])

    def valueIterationMin(self, epslon, gamma):
        normak = 1+epslon
        
        vk = numpy.zeros(len(self.S))
        vk_1 = vk

        ak = numpy.array(len(self.S))
        ak_1 = ak

       
        while(normak > epslon):
            vk=vk_1
            ak=ak_1

            vk_1 = numpy.zeros(len(self.S))
            ak_1 = numpy.zeros(len(self.S))

            for s in range(len(self.S)):
                vk_1[s], ak_1[s] = self.vMin(s, gamma, vk)

            normak = abs(vk - vk_1).max()
            print(normak)
        
        return ak

    def valueIterationMin2(self, epslon, gamma):      
        T = list()
        
        for a in range(len(self.A)):
            aux = numpy.zeros((len(self.S),len(self.S)), dtype=float)
            proximos = self.Transicoes[self.A[a]]
            for index, row in proximos.iterrows():
                self.acessosTransicoes = self.acessosTransicoes + 1
                aux[int(row['Origem'])-1][int(row['Destino'])-1] = row['Probabilidade']              
            
            T.append(aux)
        
        gc.collect()

        S = numpy.ones((len(self.S),1))
        ak, vk = self.calculaValueIteration(epslon, gamma, S, T, numpy.zeros((1,len(self.S))))

        gc.collect()
        
        return ak, vk

    def calculaValueIteration(self, epslon, gamma, S, T, v):
        normak = 1+epslon

        vk = v
        vk_1 = vk

        ak = numpy.zeros(len(self.S), dtype=int)
        ak_1 = ak

        while(normak > epslon):
            V_aux = numpy.zeros((len(self.S),len(self.A)))
            vk=vk_1
            ak=ak_1

            for a in range(len(self.A)):
                teste = self.Custo * S + (gamma*((T[a]*vk.transpose()).sum(1))).reshape((len(self.S),1))
                V_aux[:,a] = teste.reshape(len(self.S))

            vk_1 = numpy.min(V_aux, axis=1).reshape((len(self.S),1))
            ak_1 = numpy.argmin(V_aux,axis=1)

            normak = abs(vk - vk_1).max()
            print(normak)
        return ak_1, vk_1
        
    
    def vMin(self, s, gamma, vk):
        result_v = 99999999
        result_a = -1

        cost = self.custo(s)

        for a in range(len(self.A)):
            self.acessosTransicoes = self.acessosTransicoes + 1
            proximos_index = self.Transicoes[self.A[a]]['Origem'] == (s + 1)
            proximos = self.Transicoes[self.A[a]][proximos_index==True]
            
            if proximos.shape[0] > 0:
                aux = cost

                for index, row in proximos.iterrows():       
                    aux += gamma * row['Probabilidade'] * vk[int(row['Destino'])-1]
                    
                if aux < result_v:
                    result_v = aux
                    result_a = a

        return result_v, result_a

    def expandeSucessores(self, s, gs0):
        retorno = list()
        for a in range(len(self.A)):
            self.acessosTransicoes = self.acessosTransicoes + 1

            proximos_index = self.Transicoes[self.A[a]]['Origem'] == (s)
            proximos = self.Transicoes[self.A[a]][proximos_index==True]

            for index, row in proximos.iterrows():
                if int(row['Destino']) not in gs0 and int(row['Destino']) not in retorno:
                    retorno.append(int(row['Destino']))

        return retorno

    def expandeSucessores2(self, s, gs0, gsg):
        retorno = list()
        condicao_parada = False
        for a in range(len(self.A)):
            self.acessosTransicoes = self.acessosTransicoes + 1

            proximos_index = self.Transicoes[self.A[a]]['Origem'] == (s)
            proximos = self.Transicoes[self.A[a]][proximos_index==True]

            for index, row in proximos.iterrows():
                if int(row['Destino']) not in gs0 and int(row['Destino']) not in retorno:
                    if int(row['Destino']) in gsg:
                        condicao_parada = True

                    retorno.append(int(row['Destino']))

        return retorno, condicao_parada

    def expandePredecessores(self, s, gsg,expand_one, gamma, v, gs0):
        p_back = 0
        s_back = 0
        retorno = list()
        condicao_parada = False
        for a in range(len(self.A)):
            self.acessosTransicoes = self.acessosTransicoes + 1

            proximos_index = self.Transicoes[self.A[a]]['Destino'] == (s)
            proximos = self.Transicoes[self.A[a]][proximos_index==True]

            for index, row in proximos.iterrows():
                if expand_one:
                    if int(row['Origem']) != s and int(row['Origem']) not in gsg:
                        if int(row['Origem']) in gs0:
                            condicao_parada = True
                        if row['Probabilidade'] > p_back:
                            p_back = row['Probabilidade']
                            s_back = int(row['Origem'])
                else:
                    if int(row['Origem']) not in gsg and int(row['Origem']) not in retorno:
                        if int(row['Origem']) in gs0:
                            condicao_parada = True
                        retorno.append(int(row['Origem']))
        
        if expand_one and s_back > 0:
            retorno.append(s_back)

        return retorno, condicao_parada
    
    def LAO_star(self, s0, goals,epslon, gamma):
        pol = numpy.zeros(len(self.S), dtype=int)
        F = []
        F.append(s0)
        gs0 = []
        gs0.append(s0)
        gvs0 = list()
        gvs0.append(s0)
        I = list()
        F_aux = list()
        v = numpy.zeros((1,len(self.S)))
        Z = numpy.zeros((len(self.S),1))

        T = list()        
        for a in range(len(self.A)):
            aux = numpy.zeros((len(self.S),len(self.S)), dtype=float)            
            T.append(aux)

        while len(F) > 0:
            gc.collect()
            s = F.pop(0) 
            if s in gvs0:
                I.append(s) #d

                F.extend(F_aux)
                F_aux.clear()

                gs0 = []
                gs0.extend(I)
                gs0.extend(F)

                aux = self.expandeSucessores(s, gs0)

                Z[s-1] = 1

                for a in range(len(self.A)): 
                    proximos_index = self.Transicoes[self.A[a]]['Origem'] == (s)
                    proximos = self.Transicoes[self.A[a]][proximos_index==True]
                    for index, row in proximos.iterrows():
                        self.acessosTransicoes = self.acessosTransicoes + 1
                        T[a][int(row['Origem'])-1][int(row['Destino'])-1] = row['Probabilidade']

                F.extend(aux)
                gs0.extend(aux)

                #Z, T = self.montaZ(gs0, pol, s, s0)

                pol2, v2 = self.calculaValueIteration(epslon, gamma, Z, T, v)

                gvs0 = self.redefineGVS0(s0,gs0, pol2)

                print(datetime.now().strftime("%Y%m%d%H%M%S"))
                print("gvs0")
                print(gvs0)

                pol = pol2
                v = v2

                F.sort(reverse=True)
            else:
                ##se não for volta pra lista
                F_aux.append(s)
    
        return pol

    def redefineGVS0(self, s0, gs0, pol):
        new_GVS0 = list()
        buscar = list()
        buscar.append(s0)
        while len(buscar) > 0:
            s = buscar.pop(0)
            new_GVS0.append(s)
            self.acessosTransicoes = self.acessosTransicoes + 1
            proximos_index = self.Transicoes[self.A[pol[s-1]]]['Origem'] == (s)
            proximos = self.Transicoes[self.A[pol[s-1]]][proximos_index==True]

            for index, row in proximos.iterrows():
                if int(row['Destino']) in gs0 and int(row['Destino']) not in new_GVS0 and int(row['Destino']) not in buscar:
                    buscar.append(int(row['Destino']))
        
        return new_GVS0

    
    def BLAO_star(self, s0, goals, gamma, epslon, expand_one):
        pol = numpy.zeros(len(self.S), dtype=int)
        F = []
        F.append(s0)
        F2 = []
        F2.extend(goals)
        gs0 = []
        gs0.append(s0)
        gsg = []
        gsg.extend(goals)
        gvs0 = list()
        gvs0.append(s0)
        gvsg = list()
        gvsg.extend(goals)
        I = list()
        I2 = list()

        v = numpy.zeros((1,len(self.S)))
        Z = numpy.zeros((len(self.S),1))

        T = list()        
        for a in range(len(self.A)):
            aux = numpy.zeros((len(self.S),len(self.S)), dtype=float)            
            T.append(aux)


        cond = True
        at_forward = True
        at_backward = True
        expanded = False
        consistency_check = False

        while cond:
            at_forward = True
            at_backward = True
            expanded = False
            consistency_check = False
            v2 = v

            gc.collect()
            
            F_aux = list()
            while at_forward and len(F) > 0:
                s = F.pop(0) 
                if s in gvs0:
                    I.append(s) #d

                    F.extend(F_aux)
                    F_aux.clear()

                    gs0 = []
                    gs0.extend(I)
                    gs0.extend(F)

                    aux, consistency_check = self.expandeSucessores2(s, gs0, gsg)

                    Z[s-1] = 1

                    for a in range(len(self.A)): 
                        proximos_index = self.Transicoes[self.A[a]]['Origem'] == (s)
                        proximos = self.Transicoes[self.A[a]][proximos_index==True]
                        for index, row in proximos.iterrows():
                            self.acessosTransicoes = self.acessosTransicoes + 1
                            T[a][int(row['Origem'])-1][int(row['Destino'])-1] = row['Probabilidade']

                    F.extend(aux)
                    gs0.extend(aux)

                    F.sort(reverse=True)

                    at_forward = False
                    if consistency_check:
                        at_backward = False
                else:
                    ##se não for volta pra lista
                    F_aux.append(s)
                    
            while at_backward and len(F2) > 0:
                sg = F2.pop(0)
                I2.append(sg) #d
                #I.append(sg)
                #gsg.extend(I2)

                aux, consistency_check = self.expandePredecessores(sg, gsg,expand_one, gamma,v, gs0)

                F2.extend(aux)
                gsg.extend(aux)

                if len(aux) > 0:
                    expanded = True
                            
                    Z[sg-1] = 1

                    for a in range(len(self.A)): 
                        proximos_index = self.Transicoes[self.A[a]]['Origem'] == (aux[0])
                        proximos = self.Transicoes[self.A[a]][proximos_index==True]
                        for index, row in proximos.iterrows():
                            self.acessosTransicoes = self.acessosTransicoes + 1
                            T[a][int(row['Origem'])-1][int(row['Destino'])-1] = row['Probabilidade']

                at_backward = False
                if consistency_check:
                    at_backward = False

            ##reconstroi gvs0 (politica contendo estados e ações)
            gs0.extend(gsg)

            pol2, v2 = self.calculaValueIteration(epslon, gamma, Z, T, v)

            gvs0 = self.redefineGVS0(s0,gs0, pol2)

            print(datetime.now().strftime("%Y%m%d%H%M%S"))
            print("gvs0")
            print(gvs0)


            if expanded == False or len(F) == 0:
                cond = False
                for f in F:
                    if f in gvs0:
                        cond = True
            
            pol = pol2
            v = v2

        return pol2

    def redefineGVSG(self, goals, gsg, v):
        sg = goals[0]
        new_GVSG = list()
        new_GVSG.append(sg)

        buscar = list()
        buscar.append(sg)
        visitados = list()
        while len(buscar) > 0:
            s = buscar.pop(0) 
            visitados.append(s)
            dest = 0
            vDest = 99999999
            for a in range(len(self.A)):
                self.acessosTransicoes = self.acessosTransicoes + 1
                anteriores_index = self.Transicoes[self.A[a]]['Destino'] == (s)
                anteriores = self.Transicoes[self.A[a]][anteriores_index==True]

                for index, row in anteriores.iterrows():
                    if v[int(row['Origem'])-1] < vDest and int(row['Origem']) in gsg and int(row['Origem']) not in visitados:
                        dest = int(row['Origem'])
                        vDest = v[int(row['Origem'])-1]
            
            if dest > 0:
                buscar.append(dest)
                new_GVSG.append(dest)

        return new_GVSG

    def isAlcansavel(self, s1, s2, gs0):
        buscar = list()
        buscar.append(s2)
        visitados = list()
        while len(buscar) > 0:
            s = buscar.pop(0) 
            visitados.append(s)
            for a in range(len(self.A)):
                proximos_index = self.Transicoes[self.A[a]]['Origem'] == (s)
                proximos = self.Transicoes[self.A[a]][proximos_index==True]

                for index, row in proximos.iterrows():
                    if int(row['Destino']) == s1:
                        return True
                    if int(row['Destino']) not in buscar and int(row['Destino']) in gs0 and int(row['Destino']) not in visitados:
                        buscar.append(int(row['Destino']))
            
        return False

    def redefineGVS0_old(self, s0, gs0, v):
        new_GVS0 = list()
        new_GVS0.append(s0)
        new_pol = numpy.ones(len(self.S)) * -1

        buscar = list()
        buscar.append(s0)
        visitados = list()
        while len(buscar) > 0:
            s = buscar.pop(0) 
            visitados.append(s)
            dest = 0
            vDest = 99999999
            aDest = 0
            for a in range(len(self.A)):
                self.acessosTransicoes = self.acessosTransicoes + 1
                proximos_index = self.Transicoes[self.A[a]]['Origem'] == (s)
                proximos = self.Transicoes[self.A[a]][proximos_index==True]

                for index, row in proximos.iterrows():
                    if v[int(row['Destino'])-1] < vDest and int(row['Destino']) in gs0 and int(row['Destino']) not in visitados:
                        dest = int(row['Destino'])
                        aDest = a
                        vDest = v[int(row['Destino'])-1]
            
            if dest > 0:
                buscar.append(dest)
                new_GVS0.append(dest)
                new_pol[s-1] = aDest

        return new_GVS0, new_pol

    def montaZ(self, gs0, pol, no, s0):
        T = list()
        
        for a in range(len(self.A)):
            aux = numpy.zeros((len(self.S),len(self.S)), dtype=float)            
            T.append(aux)

        S = numpy.zeros((len(self.S),1))
        visitados = list()
        

        #self.isAlcansavel2(S, T, s0, pol, gs0, no, visitados) 

        #expande as ações do nó
        S[no-1] = 1
        for a in range(len(self.A)): 
            proximos_index = self.Transicoes[self.A[a]]['Origem'] == (no)
            proximos = self.Transicoes[self.A[a]][proximos_index==True]
            for index, row in proximos.iterrows():
                self.acessosTransicoes = self.acessosTransicoes + 1
                T[a][int(row['Origem'])-1][int(row['Destino'])-1] = row['Probabilidade']

        return S, T

    def isAlcansavel2(self, S, T, s, pol, gs0, no, visitados):
        visitados.append(s)
        #verifica se algum dos filhos alcança
        #proximos_index = self.Transicoes[self.A[pol[(s-1)]]]['Origem'] == (s)
        #proximos = self.Transicoes[self.A[pol[(s-1)]]][proximos_index==True]
        #for index, row in proximos.iterrows():
        #    self.acessosTransicoes = self.acessosTransicoes + 1
        #    T[pol[s-1]][int(row['Origem'])-1][int(row['Destino'])-1] = row['Probabilidade']
        #    if int(row['Destino']) in gs0 and int(row['Destino']) not in visitados:
        #        ret = self.isAlcansavel2(S,T,int(row['Destino']), pol, gs0, no, visitados)
        #        if ret == 1:
        #            S[s-1] = 1

        
        #verifica se alcança diretamente
        for a in range(len(self.A)): 
            proximos_index = self.Transicoes[self.A[a]]['Origem'] == (s)
            proximos = self.Transicoes[self.A[a]][proximos_index==True]
            for index, row in proximos.iterrows():
                self.acessosTransicoes = self.acessosTransicoes + 1
                T[a][int(row['Origem'])-1][int(row['Destino'])-1] = row['Probabilidade']
                if int(row['Destino']) == no:
                    S[s-1] = 1
                if int(row['Destino']) in gs0 and int(row['Destino']) not in visitados:
                    ret = self.isAlcansavel2(S,T,int(row['Destino']), pol, gs0, no, visitados)

                if ret == 1:
                    S[s-1] = 1
        return S[s-1]



# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:51:35 2021

@author: eloyp
"""
#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 09:38:29 2020

@author: Chris
"""


import numpy as np

import random

#import matplotlib.pyplot as plt

import networkx as nx

import math

import itertools

#import time

#import scipy.special as sps

from scipy.stats import gamma

from scipy.stats import poisson



#import pandas as pd

#with open("Par_Tot_pop_"+str(N)+"_p_"+str(p)+"_po_"+str(po)+"_tausick_"+str(tausick1)+"_taurec_"+str(taurecovered1)+"_tauinf_"+str(tauinfected1)+"_beta_"+str(beta)+"_pInitial_"+str(pininf)+"_pd_"+str(pd)+"_tsocdist_"+str(tsocdist)+"_dayduration_"+str(dayduration)+"_psocdist_"+str(psocdist)+"_posocdist_"+str(posocdist)+".csv","w") as out_file_Par:
#out_file_Par=open("Par_Tot_pop_"+str(N)+"_p_"+str(p)+"_po_"+str(po)+"_tausick_"+str(tausick1)+"_taurec_"+str(taurecovered1)+"_tauinf_"+str(tauinfected1)+"_beta_"+str(beta)+"_pInitial_"+str(pininf)+"_pd_"+str(pd)+"_tsocdist_"+str(tsocdist)+"_dayduration_"+str(dayduration)+"_psocdist_"+str(psocdist)+"_posocdist_"+str(posocdist)+".csv","w")
#parameters
#round_to_n = lambda x, n: round(x, -int(floor(log10(x))) + (n - 1))















#guaranteeing a small-world graph between people
def watts_strogatz_graph(M1, k, p1):
    # 1. Create a ring of N nodes
    G = nx.cycle_graph(M1)
    # 2. Connect each node n to k nearest neighbors

    for n in G.nodes():
        for i in range(1, k // 2 + 1):
            left = (n-i) % M1
            right = (n+i) % M1
            G.add_edge(n, left)
            G.add_edge(n, right)
            # 3. Rewire edges with probability p
    for u, v in list(G.edges()):
        if random.random() < p1:
            not_neighbors = set(G.nodes()) - set(G.neighbors(u)) - {u}
            w = random.choice(list(not_neighbors))
            G.remove_edge(u, v)
            G.add_edge(u, w)
    return G





















def Constructing(N,n2,n3,n4,n5,No,p,po,knearestneighbors,prewire):
    N2=math.ceil(N*n2)//2;
    Gr2=[]
    for i in range(N2):
        Gr2.append(nx.complete_graph(2))
        mapping={0:'F2'+str(2*i),1:'F2'+str(2*i+1)}
        Gr2[i]=nx.relabel_nodes(Gr2[i],mapping)
    
    #3 family
    N3=(math.ceil(N*n3))//3;
    Gr3=[]
    for i in range(N3):
        Gr3.append(nx.complete_graph(3))
        mapping={0:'F3'+str(3*i),1:'F3'+str(3*i+1),2:'F3'+str(3*i+2)}
        Gr3[i]=nx.relabel_nodes(Gr3[i],mapping)
    
    #4 family
    N4=(math.ceil(N*n4))//4;
    Gr4=[]
    for i in range(N4):
        Gr4.append(nx.complete_graph(4))
        mapping={0:'F4'+str(4*i),1:'F4'+str(4*i+1),2:'F4'+str(4*i+2),3:'F4'+str(4*i+3)}
        Gr4[i]=nx.relabel_nodes(Gr4[i],mapping)
    
    #institutions/buildings
    Gro=[]
    for i in range(No):
        Gro.append(nx.complete_graph(1))
        mapping={0:'o'+str(i)}
        Gro[i]=nx.relabel_nodes(Gro[i],mapping)
    
    #5 family
    N5=math.ceil(N*n5);
    Gr5=[]
    for i in range(N5):
        Gr5.append(nx.complete_graph(5))
        mapping={0:'F5'+str(5*i),1:'F5'+str(5*i+1),2:'F5'+str(5*i+2),3:'F5'+str(5*i+3),4:'F5'+str(5*i+4)}
        Gr5[i]=nx.relabel_nodes(Gr5[i],mapping)
    
    #1 family
    N1=N-5*N5-2*N2-3*N3-4*N4-No;
    Gr1=[]
    for i in range(N1):
        Gr1.append(nx.complete_graph(1))
        mapping={0:'F1'+str(i)}
        Gr1[i]=nx.relabel_nodes(Gr1[i],mapping)


        #initializing list with all population
    pop=[]
    for i in range(N1):
        pop.append(Gr1[i].nodes())
    
    for i in range(N2):
        pop.append(Gr2[i].nodes())
        
    for i in range(N3):
        pop.append(Gr3[i].nodes())
        
    for i in range(N4):
        pop.append(Gr4[i].nodes())
        
    for i in range(N5):
        pop.append(Gr5[i].nodes())
    
    popmerged = list(itertools.chain.from_iterable(pop)) #flattens
    Gpop=nx.Graph()

    for i in range(len(popmerged)):
        Gpop.add_node(popmerged[i])


    NtotH=N1+N2+N3+N4+N5 #total number of houses
    GSW = watts_strogatz_graph(NtotH, knearestneighbors, prewire) #construct the Small-World network
    Glist=[Gr1, Gr2, Gr3, Gr4, Gr5]
    nlista=list(itertools.chain.from_iterable(Glist))
    choicehouse=np.random.choice(len(nlista), len(nlista), replace=False)
    nlistmerged=[]
    for qw in range(len(choicehouse)):
        nlistmerged.append(nlista[choicehouse[qw]])
    
    totedgesGSWpresent=math.ceil(len(GSW.edges())*p) #Decide how many edges of SW network stay
    GSWcopy=GSW.copy()
    
    if totedgesGSWpresent>0:
        for i in range(totedgesGSWpresent):
            s1=random.randint(0,NtotH-1) 
            s2=random.randint(0,NtotH-1)
            while GSWcopy.has_edge(s1,s2)==False: #select two houses at random that are connected in GSW
                s1=random.randint(0,NtotH-1)
                s2=random.randint(0,NtotH-1)
            ss1=random.randint(0,len(nlistmerged[s1].nodes())-1) #select members from each house
            ss2=random.randint(0,len(nlistmerged[s2].nodes())-1)
            Gpop.add_edge(nlistmerged[s1].nodes()[ss1],nlistmerged[s2].nodes()[ss2])
            GSWcopy.remove_edge(s1,s2) # to avoid repetition
    
    #generating connection between other (institutions/buildings) and population
    Go=nx.Graph()

    Gromerged= list(itertools.chain.from_iterable(Gro))       
    for j in range(len(Gro)):
        for i in range(len(popmerged)):
                r = random.random()
                if r < po:
                    Go.add_edge(Gromerged[j],popmerged[i])
    
                
    #generating total network
    GTot=nx.Graph()
    GTot.add_edges_from(Go.edges())
    GTot.add_edges_from(Gpop.edges())
    for i in range(N1):
        GTot.add_nodes_from(Gr1[i].nodes())
    
    for i in range(N2):
        GTot.add_edges_from(Gr2[i].edges())
        
    for i in range(N3):
        GTot.add_edges_from(Gr3[i].edges())
        
    for i in range(N4):
        GTot.add_edges_from(Gr4[i].edges())
        
    for i in range(N5):
        GTot.add_edges_from(Gr5[i].edges())
    
    #prob. of being hospitalized, prob. of being recovering, prob. of being infected
    GTot.add_node('o1')

    return GTot,Gromerged,popmerged,Go,Gpop,Gr1,Gr2,Gr3,Gr4,Gr5,nlistmerged,GSW













def initializing(GTot,GTot1,Go,Gpop,pininf,N,popmerged,alphatemp,betatemp,R01,alphasick,betasick,alpharecovered,betarecovered,alphainfected,betainfected,alphahosp,betahosp,alpharec,betarec,resuming,cluster):
    #created this list because was not sure how to access name of node
    lista=GTot1.nodes()
    
    #setting default of state of population as susceptible
    for i in range(len(lista)):
        if lista[i].find('F')==0:
            d=lista[i]
            GTot1.node[d]['state']='susceptible'
        else:
            d=lista[i]
            GTot1.node[d]['state']='other'
    
    totinitialinf=math.ceil(pininf*N)
    rilist=[]
    for i in range(totinitialinf):
        ri=random.randint(0,len(popmerged)-1)
        if ri in rilist:
            i-=1
        else:
            rilist.append(ri)
            
    out_string_Sim1=""
    
    if resuming==0:        
        for j1 in range(len(rilist)):
            initialinfected=popmerged[rilist[j1]]
            for i in range(len(lista)):
            #generate initial state
                if (lista[i].find('F')==0) and (lista[i]==initialinfected):
                        #checkinfect+=1
                        if cluster==0:
                            print('initial patient', initialinfected)
                        out_string_Sim1+=str(0)
                        out_string_Sim1+=","+"initial patient "+str(initialinfected)
                        out_string_Sim1+="\n"
                        d=lista[i]
                        GTot.node[d]['state']='infected'
                        GTot1.node[d]['state']='infected'
                        GTot1.node[d]['init time infected']=0
                        GTot1.node[d]['color']='grey'
                        GTot1.node[d]['importance']=100
                        GTot1.node[d]['R0']=R01
                        for k in GTot1.neighbors(d):
                            if ((GTot1.node[k]).get('state')).find('h')!=0:
                                if ((GTot1.node[k]).get('state')).find('r')!=0:
                                    GTot1[d][k]['color']='grey'
                                    GTot1[d][k]['size']=4
                                    if k.find('ο')!=0:
                                        for q in GTot1.neighbors(k):
                                            GTot1[q][k]['color']='grey'
                                            GTot1[q][k]['size']=4
        
    
    for i in range(len(lista)):
        if lista[i].find('o')==0:
            d=lista[i]
            GTot1.node[d]['state']='other'
            GTot1.node[d]['social temp']=random.gammavariate(alphatemp[1],betatemp[1])
            if cluster==0:
                print('social temp ', GTot1.node[d]['social temp'])
    
    
    GTot1.node['o1']['social temp']=random.gammavariate(alphatemp[2],betatemp[2])
    #initializing time spend connected
    for t in GTot1.edges():
        GTot1[t[0]][t[1]]['time']=random.random()
    
    #graph of people in houses (no connections between houses neither between houses and places)
    GTot2=GTot1.copy()
    GTot2.remove_edges_from(Go.edges())
    GTot2.remove_edges_from(Gpop.edges())
    
    for i in GTot2.edges():
        GTot1[i[0]][i[1]]['time']=1
        
        
    infected1=[v for v in GTot1.node if GTot1.node[v]['state'] == 'infected']
    susceptible1=[v for v in GTot1.node if GTot1.node[v]['state'] == 'susceptible']
    other1=[v for v in GTot1.node if GTot1.node[v]['state'] == 'other']

        
    for i in infected1:
        GTot1.node[i]['spin']=1
        GTot1.node[i]['detected']='no'
        GTot1.node[i]['viral load']=1/4+random.random()/4
        GTot1.node[i]['init time infected']=0
        GTot1.node[i]['init time sick']=0
        GTot1.node[i]['init time recovered']=0
        GTot1.node[i]['init time hosp']=0
        GTot1.node[i]['init time rec']=0
        GTot1.node[i]['num of neigh']=0
        GTot1.node[i]['num of neigh det']=0
        GTot1.node[i]['R0']=R01
        
        
    for i in susceptible1:
        GTot1.node[i]['spin']=-1
        GTot1.node[i]['detected']='no'
        GTot1.node[i]['viral load']=0
        GTot1.node[i]['init time recovered']=0
        GTot1.node[i]['init time infected']=0
        GTot1.node[i]['init time sick']=0
        GTot1.node[i]['init time hosp']=0
        GTot1.node[i]['init time rec']=0
        GTot1.node[i]['num of neigh']=0
        GTot1.node[i]['num of neigh det']=0
        GTot1.node[i]['R0']=R01
        
        
    for i in other1:
        GTot1.node[i]['spin']=0
        GTot1.node[i]['viral load']=1/2+random.random()/2
        GTot1.node[i]['detected']='no'
        
        
    for i in GTot1.nodes():
        if i.find('F')==0:
            if random.random()<0.5:
                GTot1.node[i]['gender']='female'
            else:
                GTot1.node[i]['gender']='male'
            GTot1.node[i]['age']=np.random.choice(10,p=[0.09, 0.1, 0.1, 0.13, 0.17, 0.16, 0.11, 0.08, 0.05, 0.01])
        else:
            GTot1.node[i]['gender']='does not apply'
            GTot1.node[i]['age']='does not apply'
    for i in GTot1.nodes():  
        GTot1.node[i]['tausick']=random.gammavariate(alphasick,betasick)
        GTot1.node[i]['taurecovered']=random.gammavariate(alpharecovered,betarecovered)
        GTot1.node[i]['tauinfected']=random.gammavariate(alphainfected,betainfected)
        GTot1.node[i]['tauhosp']=random.gammavariate(alphahosp,betahosp)
        GTot1.node[i]['taurec']=random.gammavariate(alpharec,betarec)

        
    return GTot1,out_string_Sim1
        
            
        
        
        
        
        
        
        
        
        

def init_plot(GTot1,GTot,R01,alphatemp,betatemp,resuming,cluster):
    lista=GTot1.nodes()
    for i in range(len(lista)):
        d=lista[i]
        if (lista[i].find('F')==0) and (((GTot1.node[d]).get('state')).find('i')!=0):
            GTot.node[d]['state']='susceptible'
            GTot1.node[d]['state']='susceptible'
            GTot1.node[d]['color']='orange'
            GTot1.node[d]['importance']=100
            GTot1.node[d]['R0']=R01
            for k in GTot1.neighbors(d):
                if ((GTot1.node[k]).get('state')).find('h')!=0: 
                    if ((GTot1.node[k]).get('state')).find('i')!=0:
                        GTot1[d][k]['color']='orange'
                        GTot1[d][k]['size']=0.5

    #take care of nodes colors and weights
    for i in range(len(lista)):
        if lista[i].find('o')==0:
            d=lista[i]
            GTot.node[d]['color']='blue'
            GTot.node[d]['importance']=1000
            GTot.node[d]['state']='other'
            GTot1.node[d]['color']='blue'
            GTot1.node[d]['importance']=1000
            GTot1.node[d]['state']='other'
            GTot1.node[d]['social temp']=random.gammavariate(alphatemp[1],betatemp[1])
            if cluster==0:
                print('social temp ', GTot1.node[d]['social temp'])
            if lista[i].find('o1')==0:
                GTot.node[d]['color']='red'
                GTot1.node[d]['color']='red'
        elif lista[i].find('F1')==0:
            d=lista[i]
            GTot.node[d]['color']='green'
            GTot.node[d]['importance']=100
        elif lista[i].find('F2')==0:
            d=lista[i]
            GTot.node[d]['color']='yellow'
            GTot.node[d]['importance']=100
        elif lista[i].find('F3')==0:
            d=lista[i]
            GTot.node[d]['color']='red'
            GTot.node[d]['importance']=100
        elif lista[i].find('F4')==0:
            d=lista[i]
            GTot.node[d]['color']='purple'
            GTot.node[d]['importance']=100
        elif lista[i].find('F5')==0:
            d=lista[i]
            GTot.node[d]['color']='orange'
            GTot.node[d]['importance']=100
        else:
            d=lista[i]
            GTot.node[d]['color']='red'
            
    return GTot,GTot1












def probIJ(GTot1,betafree):
    spincand=[]
    pij=[]
    Gtij=nx.Graph()
    for i in GTot1.nodes():
        if (GTot1.node[i]['state'] == 'susceptible'):
            neig=GTot1.neighbors(i)
            for j in neig:
                if j in GTot1.nodes():
                    tij=0
                    if (GTot1.node[j]['state'] == 'infected') or (GTot1.node[j]['state'] == 'hospitalized'):
                        tij=0
                        if (GTot1.has_edge(i,j)):
                            tij=GTot1[i][j]['time']*GTot1.node[j]['viral load']
                            Gtij.add_edge(i,j)
                            Gtij[i][j]['prob']=betafree*tij
                             
            #neig=GTot1.neighbors(i)
            #for k in neig:
                    neigneig=GTot1.neighbors(j)
                    for k1 in neigneig:
                        if k1 in GTot1.nodes():
                            if (j.find('o')==0) and ((GTot1.node[k1]['state'] == 'infected') or (GTot1.node[k1]['state'] == 'hospitalized')):
                                tij=tij+GTot1.node[j]['social temp']*GTot1[i][j]['time']*GTot1.node[i]['viral load']*GTot1[j][k1]['time']*GTot1.node[j]['viral load']
                                Gtij.add_edge(i,k1)
                                Gtij[i][k1]['prob']=tij
        
    for a in Gtij.nodes():
        if (GTot1.node[a]['state'] == 'susceptible'):
            spincand.append(a)
            nb=Gtij.neighbors(a)
            suma=0
            for j in nb:
                if (GTot1.node[j]['R0']>0):
                    Jij=Gtij[a][j]['prob']
                    sij=Jij*GTot1.node[j]['spin']
                    suma=suma+sij
                    
            if suma>0:
                pij.append(0.5*(1-GTot1.node[a]['spin']*np.tanh(suma)))
            else:
                pij.append(0)
                
    return Gtij,spincand,pij






def probNM(GTot1,pij,spincand,dayduration,t,alphasick,betasick,alpharecovered,betarecovered,alphainfected,betainfected,alphahosp,betahosp,alpharec,betarec,hospitalized1,infected1,recovered1,susceptible1,other1,cluster):
    

    lambdas=[]
    proctyp=[]
    changedperson=[]

    
    sumi=0
    for qqq in GTot1.nodes():
        if qqq in spincand:
            psicontagion=poisson.pmf(int((1/(pij[spincand.index(qqq)]-0.5))*dayduration),int((1/(pij[spincand.index(qqq)]-0.5))*dayduration))
            psicdfcontagion=1-poisson.cdf(int((1/(pij[spincand.index(qqq)]-0.5))*dayduration),int((1/(pij[spincand.index(qqq)]-0.5)*dayduration)))

#            if cluster==0:
#                print('psicontagion ', psicontagion)
#                print('psicdfcontagion ', psicdfcontagion)
#                print('lambdacontagion', psicontagion/psicdfcontagion)
            lambdas.append(psicontagion/psicdfcontagion)

            proctyp.append(0)
            changedperson.append(qqq)

            if math.isnan(psicontagion/psicdfcontagion):
                sumi=sumi
            else:
                sumi=sumi+psicontagion/psicdfcontagion
        if qqq in hospitalized1:
            psisick=gamma.pdf(t-GTot1.node[qqq]['init time sick'],alphasick,scale=betasick)
            psicdfsick=1-gamma.cdf(t-GTot1.node[qqq]['init time sick'],alphasick,scale=betasick)
#            print("Primer denominador este raro psicdfsick", psicdfsick)
            lambdas.append(psisick/psicdfsick)
            proctyp.append(2)
            changedperson.append(qqq)
            if math.isnan(psisick/psicdfsick):
                sumi=sumi
            else:
                sumi=sumi+psisick/psicdfsick
        if qqq in recovered1:
            psirecovered=gamma.pdf(t-GTot1.node[qqq]['init time recovered'],alpharecovered,scale=betarecovered)
            psicdfrecovered=1-gamma.cdf(t-GTot1.node[qqq]['init time recovered'],alpharecovered,scale=betarecovered)
#            print("Segundo denominador raro psicdfrecovered", psicdfrecovered)
            lambdas.append(psirecovered/psicdfrecovered)
            proctyp.append(3)
            changedperson.append(qqq)
            if math.isnan(psirecovered/psicdfrecovered):
                sumi=sumi
            else:
                sumi=sumi+psirecovered/psicdfrecovered
        if qqq in infected1:
            psiinfected=gamma.pdf(t-GTot1.node[qqq]['init time infected'],alphainfected,scale=betainfected)
            psicdfinfected=1-gamma.cdf(t-GTot1.node[qqq]['init time infected'],alphainfected,scale=betainfected)
#            print("Tercer denominador raro psicdfinfected", psicdfinfected)            
            lambdas.append(psiinfected/psicdfinfected)
            proctyp.append(5)
            changedperson.append(qqq)
            if math.isnan(psiinfected/psicdfinfected):
                sumi=sumi
            else:
                sumi=sumi+psiinfected/psicdfinfected
        if qqq in infected1:
            psihosp=gamma.pdf(t-GTot1.node[qqq]['init time infected'],alphahosp,scale=betahosp)
            psicdfhosp=1-gamma.cdf(t-GTot1.node[qqq]['init time infected'],alphahosp,scale=betahosp)
#            if cluster==0:
#                print('psihosp ', psihosp)
#                print('psicdfhosp ', psicdfhosp)
#            print("Cuarto denominador raro psicdfhosp", psicdfhosp)   
            lambdas.append(psihosp/psicdfhosp)
            proctyp.append(4)
            changedperson.append(qqq)
            #sumi=sumi+psihosp/psicdfhosp
            if math.isnan(psihosp/psicdfhosp):
                sumi=sumi
            else:
                sumi=sumi+psihosp/psicdfhosp
        if qqq in hospitalized1:
            psirec=gamma.pdf(t-GTot1.node[qqq]['init time sick'],alpharec,scale=betarec)
            psicdfrec=1-gamma.cdf(t-GTot1.node[qqq]['init time sick'],alpharec,scale=betarec)
#            print("Quinto denominador raro psicdfrec", psicdfrec)
            lambdas.append(psirec/psicdfrec)
            proctyp.append(1)
            changedperson.append(qqq)
            if math.isnan(psirec/psicdfrec):
                sumi=sumi
            else:
                sumi=sumi+psirec/psicdfrec
        
    lambdas1=[0 if math.isnan(x) else x for x in lambdas]
    
#    if (len(spincand)+len(recovered1)+2*(len(hospitalized1)+len(infected1))) == 0:
# =============================================================================
#         print("EL DENOMINADOR ES 0")
#         print("spincand", spincand)
#         print("recovered1", recovered1)
#         print("hospitalized1", hospitalized1)
#         print("infected1", infected1)
#         print("sumi", sumi)
# =============================================================================
        
#    if sumi == 0:
# =============================================================================
#         print("EL DENOMINADOR ES 0 POR SUMI")
#         print("spincand", spincand)
#         print("recovered1", recovered1)
#         print("hospitalized1", hospitalized1)
#         print("infected1", infected1)
#         print("sumi", sumi)
# =============================================================================
        
    totproc=[1/(len(spincand)+len(recovered1)+2*(len(hospitalized1)+len(infected1))) if math.isnan(qqq/(sumi)) else qqq/(sumi) for qqq in lambdas1]
    
#    print(sum(totproc))
#    if (sum(totproc)>1) and (sum(totproc)-1<10**-4):
#        print('hola1')
#        totproc[len(totproc)-1]-=(sum(totproc)-1)
#    if (sum(totproc)<1) and (1-sum(totproc)<10**-4):
#        print('hola2')
#        totproc[len(totproc)-1]+=(1-sum(totproc))
#    
#    
#    
#    print(sum(totproc))
#    print(sum([1/(len(lambdas1)) for qqq in lambdas1]))
#    print(sum([qqq/(sumi) for qqq in lambdas1]))

    if sumi>0:
        tottim=-np.log(random.random())/(sumi)
    else:
        tottim=0
    return sumi,proctyp,changedperson,totproc,tottim
    
    
    
    
    
    
    
    
def reloading(sume,tsocdist,tsocdistlift,mobility,resuming,runn,cluster,outfilename):
    dataGraphnodes=np.loadtxt(outfilename+"/Current_sume_"+str(sume)+"_tsocdist_"+str(tsocdist)+"_tsocdistlift_"+str(tsocdistlift)+"_mobility_"+str(mobility)+"_resuming_"+str(resuming)+"_runn_"+str(runn)+".csv",dtype=str,delimiter=',',comments='#')
    t=int(dataGraphnodes[0][0])
#    if cluster==0:
#        print('resuming time: ',t)
    GTots=nx.Graph()
    for j in range(len(dataGraphnodes)):
        GTots.add_node(dataGraphnodes[j][1])
    ss=0
    for q in GTots.nodes():
        GTots.node[q]['viral load']=float(dataGraphnodes[ss][6])
        GTots.node[q]['detected']=dataGraphnodes[ss][7]
        GTots.node[q]['spin']=float(dataGraphnodes[ss][8])
        GTots.node[q]['state']=dataGraphnodes[ss][9]
        GTots.node[q]['gender']=dataGraphnodes[ss][10]
        if q.find('o')==0:
            GTots.node[q]['age']=dataGraphnodes[ss][11]
        else:
            GTots.node[q]['R0']=int(dataGraphnodes[ss][2])
            GTots.node[q]['age']=int(dataGraphnodes[ss][11])
            GTots.node[q]['init time sick']=float(dataGraphnodes[ss][3])
            GTots.node[q]['init time infected']=float(dataGraphnodes[ss][4])
            GTots.node[q]['init time recovered']=float(dataGraphnodes[ss][5])
            GTots.node[q]['num of neigh']=int(dataGraphnodes[ss][16])
            GTots.node[q]['num of neigh det']=int(dataGraphnodes[ss][17])
            for q2 in range(GTots.node[q]['num of neigh']):
                GTots.node[q]['neigh'+str(q2)]=dataGraphnodes[ss][17+1+q2]
            for q2 in range(GTots.node[q]['num of neigh det']):
                GTots.node[q]['neigh det'+str(q2)]=dataGraphnodes[ss][17+GTots.node[q]['num of neigh']+1+q2]
        GTots.node[q]['tausick']=float(dataGraphnodes[ss][12])
        GTots.node[q]['taurecovered']=float(dataGraphnodes[ss][13])
        GTots.node[q]['tauinfected']=float(dataGraphnodes[ss][14])
        if q.find('o')==0:
            GTots.node[q]['social temp']=float(dataGraphnodes[ss][15])
        
        ss+=1

    GTotp=nx.read_edgelist("Current_edges_sume"+str(sume)+"_tsocdist_"+str(tsocdist)+"_tsocdistlift_"+str(tsocdistlift)+"_mobility_"+str(mobility)+"_resuming_"+str(resuming)+"_runn_"+str(runn)+".csv")

    GTots.add_edges_from(GTotp.edges(data=True))

    GTot1=nx.Graph()
    GTot1=GTots.copy()
    
    GTotpInit=nx.read_edgelist("init_edges_sume"+str(sume)+"_tsocdist_"+str(tsocdist)+"_tsocdistlift_"+str(tsocdistlift)+"_mobility_"+str(mobility)+"_resuming_"+str(resuming)+"_runn_"+str(runn)+".csv")

    GTotsInit=nx.Graph()
    GTotsInit.add_edges_from(GTotpInit.edges(data=True))
    
    GTot4=GTotsInit.copy()
    
    
    return GTot1,GTot4
    
    
    
    
    
    
    
    
    
    
def social_dist(t,Gpop,psocdist,GTot1,Go,posocdist,quarantine,cluster):
    #out_string_Sim1=""
    out_string_Sim1=[]
    for j in Gpop.edges():
        if random.random()<psocdist:
            if (GTot1.has_edge(j[0],j[1])==True):
                    GTot1.remove_edge(j[0],j[1])
                    Gpop.remove_edge(j[0],j[1])
#                    if cluster==0:
#                        print('edge',j[0],'to',j[1],'was removed')
                    #out_string_Sim1+=str(t)
                    out_string_Sim1.append(str(t))
                    #out_string_Sim1+=","+"edge"+str(j[0])+"to"+str(j[1])+" was removed"
                    out_string_Sim1.append("edge"+str(j[0])+"to"+str(j[1])+" was removed")
                    #out_string_Sim1+="\n"
            else:
                if (((GTot1.node[j[0]]).get('state')).find('h')==0):
                    for k in range(GTot1.node[j[0]]['num of neigh']):
                        d=GTot1.node[j[0]]['neigh'+str(k)]
                        if  d==j[1]:
                            #out_string_Sim1+=str(t)
                            out_string_Sim1.append(str(t))
                            #out_string_Sim1+=","+"edge"+str(j[0])+"to"+str(j[1])+" was removed"
                            out_string_Sim1.append("edge"+str(j[0])+"to"+str(j[1])+" was removed")
                            #out_string_Sim1+="\n"
                            GTot1.node[j[0]]['neigh'+str(k)]=[]
                            Gpop.remove_edge(j[0],j[1])
                elif (((GTot1.node[j[1]]).get('state')).find('h')==0):
                    for k in range(GTot1.node[j[1]]['num of neigh']):
                        d=GTot1.node[j[1]]['neigh'+str(k)]
                        if  d==j[0]:
                            #out_string_Sim1+=str(t)
                            out_string_Sim1.append(str(t))
                            #out_string_Sim1+=","+"edge"+str(j[0])+"to"+str(j[1])+" was removed"
                            out_string_Sim1.append("edge"+str(j[0])+"to"+str(j[1])+" was removed")
                            #out_string_Sim1+="\n"
                            GTot1.node[j[1]]['neigh'+str(k)]=[]
                            Gpop.remove_edge(j[0],j[1])

                    
    
    for j in Go.edges():
        rosocdist=random.random()
        if rosocdist<posocdist:
            if (((GTot1.node[j[0]]).get('state')).find('h')!=0) and (((GTot1.node[j[1]]).get('state')).find('h')!=0):
                if (GTot1.has_edge(j[0],j[1])==True):
                    GTot1.remove_edge(j[0],j[1])
                    Go.remove_edge(j[0],j[1])
#                    if cluster==0:
#                        print('edge',j[0],'to',j[1],'was removed')
                    #out_string_Sim1+=str(t)
                    out_string_Sim1.append(str(t))
                    #out_string_Sim1+=","+"edge"+str(j[0])+"to"+str(j[1])+" was removed"
                    out_string_Sim1.append("edge"+str(j[0])+"to"+str(j[1])+" was removed")
                    #out_string_Sim1+="\n"
                elif quarantine==1: 
                    if (GTot1.node[j[0]]['detected']=='yes'):
                        for k in range(GTot1.node[j[0]]['num of neigh']):
                            d=GTot1.node[j[0]]['neigh'+str(k)]
                            if  d==j[1]:
                                GTot1.node[j[0]]['neigh'+str(k)]=[]
                                Go.remove_edge(j[0],j[1])
#                                if cluster==0:
#                                    print('edge',j[0],'to',j[1],'was removed')
                                #out_string_Sim1+=str(t)
                                out_string_Sim1.append(str(t))
                                #out_string_Sim1+=","+"edge"+str(j[0])+"to"+str(j[1])+" was removed"
                                out_string_Sim1.append(str(j[0])+"to"+str(j[1])+" was removed")
                                #out_string_Sim1+="\n"
                                
    return GTot1,Go,Gpop,out_string_Sim1








def social_dist_lifting(t,GTot1,GTot4,pmobilityafter,cluster,pmobility,quarantine,bt):
    #out_string_Sim1=""
    out_string_Sim1=[]
    stilldetected=0
    lifting=0
    #pmobility=pmobility
    #pmobility=pmobilitysocdist
    #quarantine=1
    for q in GTot1.nodes():
        if (GTot1.node[q]['detected']=='yes') and (((GTot1.node[q]).get('state')).find('i')==0):
            stilldetected=1         
    if stilldetected==0:
        lifting=1
        bt=2*3
#        if cluster==0:
#            print('social distancing and quarantine are lifted at time: ',t)
        quarantine=0
        pmobility=pmobilityafter
        for q in GTot4.edges():
            if (GTot1.has_node(q[0])==True) and (GTot1.has_node(q[1])==True):
                if GTot1.has_edge(q[0],q[1])==False:
                    GTot1.add_edge(q[0],q[1])
#                    if cluster==0:
#                        print('edge',q[0],'to',q[1],'was recovered')
                    time1=random.random()
                    GTot1[q[0]][q[1]]['time']=time1
                    #out_string_Sim1+=str(t)
                    out_string_Sim1.append(str(t))
                    #out_string_Sim1+=","+"edge"+str(q[0])+"to"+str(q[1])+" was recovered"
                    out_string_Sim1.append("edge"+str(q[0])+"to"+str(q[1])+" was recovered")
                    #out_string_Sim1+="\n"
    
    return GTot1,pmobility,stilldetected,quarantine,bt,lifting,out_string_Sim1
                    
                    








def proc_infection(spincand,pij,GTot1,quarantine,pd,newinfdet,a,t,newinf,R01,Gtij,cluster):
    #out_string_Sim1=""
    out_string_Sim1=[]
    update=0
    for qqq in spincand:
        if a==qqq:
            pij1=pij[spincand.index(qqq)]

            if (random.random() < pij1):# and (flipped==0):
                update=1
#                if cluster==0:
#                    print('time:',t)
                #out_string_Sim1=str(t)
                out_string_Sim1.append(str(t))
                nb=Gtij.neighbors(a)
                for j in nb:
                    GTot1.node[j]['R0']=GTot1.node[j]['R0']-1

                GTot1.node[a]['spin']*=-1
                GTot1.node[a]['state'] = 'infected'
                GTot1.node[a]['init time infected']=t
                rd=random.random()
                if quarantine==1:
                    if rd<pd:
                        newinfdet+=1
                        GTot1.node[a]['detected']='yes'
                        GTot1.node[a]['num of neigh det']=len(GTot1.neighbors(a))
                        q1=0
                        for k in GTot1.neighbors(a):
                            GTot1.node[a]['neigh det'+str(q1)]=k
                            q1+=1
                            GTot1.remove_edge(a,k)
#                        if cluster==0:
#                            print(a,'infected and detected')
                        #out_string_Sim1+=","+str(a)+" infected and detected"
                        out_string_Sim1.append(str(a)+" infected and detected")
                        #out_string_Sim1+="\n"
                    else:
                        GTot1.node[a]['detected']='no'
#                        if cluster==0:
#                            print(a,'infected but not detected')
                        #out_string_Sim1+=","+str(a)+" infected but not detected "
                        out_string_Sim1.append(str(a)+" infected but not detected ")
                        #out_string_Sim1+="\n"
                else: 
#                    if cluster==0:
#                        print(a,'infected but not detected')
                    #out_string_Sim1+=","+str(a)+" infected but not detected "
                    out_string_Sim1.append(str(a)+" infected but not detected ")
                    #out_string_Sim1+="\n"
                newinf+=1
                GTot1.node[a]['R0']=R01
        
    return out_string_Sim1,GTot1,update,newinfdet,newinf
                
                
                
                
                
                
                
                
                

def proc_rec_hosp(GTot1,i,t,newrec,quarantine,newrecdet,cluster):
    #out_string_Sim1=""
    out_string_Sim1=[]
#    if cluster==0:
#        print('time:',t)
#        print(i,'recovered from hospital')
    #out_string_Sim1+=str(t)
    out_string_Sim1.append(str(t))
    GTot1.node[i]['state'] = 'recovered'
    GTot1.node[i]['init time recovered']=t
    newrec+=1
    #out_string_Sim1+=","+str(i)+" recovered from hospital"
    out_string_Sim1.append(str(i)+" recovered from hospital")
    #out_string_Sim1+="\n"
    GTot1.node[i]['spin']=-1

    GTot1.remove_edge(i,'o1')
    if (quarantine==1) and (GTot1.node[i]['detected']=='yes'):
        newrecdet+=1
        for k in range(GTot1.node[i]['num of neigh det']):
            d=GTot1.node[i]['neigh det'+str(k)]
            if (GTot1.has_node(d)==True) and (((GTot1.node[d]).get('state')).find('h')!=0):
                if (((GTot1.node[d]).get('state')).find('i')!=0):
                    GTot1.add_edge(i,d)
                    GTot1[i][d]['time']=random.random()
                elif (GTot1.node[d]['detected']=='no'):
                    GTot1.add_edge(i,d)
                    GTot1[i][d]['time']=random.random()
            GTot1.node[i]['neigh det'+str(k)]=[]
        GTot1.node[i]['num of neigh det']=0
    else:
        for k in range(GTot1.node[i]['num of neigh']):
            d=GTot1.node[i]['neigh'+str(k)]
            if (GTot1.has_node(d)==True) and (((GTot1.node[d]).get('state')).find('h')!=0):
                GTot1.add_edge(i,d)
                GTot1[i][d]['time']=random.random()
            GTot1.node[i]['neigh'+str(k)]=[]
        GTot1.node[i]['num of neigh']=0
    GTot1.node[i]['detected']=='no'
    
    return GTot1,out_string_Sim1
    
    
    


def proc_hosp(quarantine,GTot1,newhosp,i,t,cluster):
    update=0
    #out_string_Sim1=""
    out_string_Sim1=[]
    if quarantine==1:
        if (GTot1.node[i]['detected']=='yes'):
            update=1
#            if cluster==0:
#                print('time:',t)
#                print(i,'hospitalized')
            #out_string_Sim1+=str(t)
            out_string_Sim1.append(str(t))
            GTot1.node[i]['state']='hospitalized'
            GTot1.node[i]['init time sick']=t
            newhosp+=1
            #out_string_Sim1+=","+str(i)+" hospitalized from detected"
            out_string_Sim1.append(str(i)+" hospitalized from detected")
            #out_string_Sim1+="\n"
            GTot1.add_edge(i,'o1')
            GTot1[i]['o1']['time']=1

    else:
        update=1
#        if cluster==0:
#            print('time:',t)
#            print(i,'hospitalized')
        #out_string_Sim1+=str(t)
        out_string_Sim1.append(str(t))
        GTot1.node[i]['state']='hospitalized'
        GTot1.node[i]['init time sick']=t
        newhosp+=1
        #out_string_Sim1+=","+str(i)+" hospitalized no quarantine"
        out_string_Sim1.append(str(i)+" hospitalized no quarantine")
        #out_string_Sim1+="\n"
    GTot1.node[i]['num of neigh']=len(GTot1.neighbors(i))
    q1=0
    for k in GTot1.neighbors(i):
        GTot1.node[i]['neigh'+str(q1)]=k
        q1+=1
        GTot1.remove_edge(i,k)
    GTot1.add_edge(i,'o1')
    GTot1[i]['o1']['time']=1
    
    return GTot1,update,out_string_Sim1,newhosp







def proc_rec_inf(GTot1,t,i,quarantine,newrecdet,newrec,cluster):
    #out_string_Sim1=""
    out_string_Sim1=[]
#    if cluster==0:
        #print((t-GTot1.node[i]['init time infected']))
        #print(GTot1.node[i]['tauinfected'])
#        print('time:',t)
    update=1
    #out_string_Sim1+=str(t)
    out_string_Sim1.append(str(t))
    GTot1.node[i]['state'] = 'recovered'
    GTot1.node[i]['init time recovered']=t
    if quarantine==1:    
        if (GTot1.node[i]['detected']=='yes'):
            for k in range(GTot1.node[i]['num of neigh det']):
                d=GTot1.node[i]['neigh det'+str(k)]
                if (GTot1.has_node(d)==True) and (((GTot1.node[d]).get('state')).find('h')!=0):
                    if (((GTot1.node[d]).get('state')).find('i')!=0):
                        GTot1.add_edge(i,d)
                        GTot1[i][d]['time']=random.random()
                    elif (GTot1.node[d]['detected']=='no'):
                        GTot1.add_edge(i,d)
                        GTot1[i][d]['time']=random.random()
                GTot1.node[i]['neigh det'+str(k)]=[]
            GTot1.node[i]['num of neigh det']=0
            newrecdet+=1
#            if cluster==0:
#                print(i,'recovered from infected detected')
            #out_string_Sim1+=","+str(i)+" recovered from infected detected"
            out_string_Sim1.append(str(i)+" recovered from infected detected")
            #out_string_Sim1+="\n"
        else:
            newrec+=1
#            if cluster==0:
#                print(i,'recovered from infected not detected')
            #out_string_Sim1+=","+str(i)+" recovered from infected not detected"
            out_string_Sim1.append(str(i)+" recovered from infected not detected")
            #out_string_Sim1+="\n"
    else:
        update=1
        newrec+=1
#        if cluster==0:
#            print(i,'recovered from infected not detected')
        #out_string_Sim1+=","+str(i)+" recovered from infected not detected"
        out_string_Sim1.append(str(i)+" recovered from infected not detected")
        #out_string_Sim1+="\n"
    GTot1.node[i]['spin']=-1
    GTot1.node[i]['detected']=='no'

    return GTot1,update,out_string_Sim1,newrec,newrecdet



    
    
def plotting(GTot1):
    #determining color of nodes
    for q in GTot1.nodes():
        if q.find('o')==0:
            GTot1.node[q]['importance']=400
            GTot1.node[q]['color']='blue'
            if q.find('o1')==0:
                GTot1.node[q]['color']='red'
                GTot1.node[q]['importance']=400
        else:
            GTot1.node[q]['importance']=100
            if ((GTot1.node[q]).get('state')).find('s')==0:
                GTot1.node[q]['color']='orange'
            if ((GTot1.node[q]).get('state')).find('r')==0:
                GTot1.node[q]['color']='green' 
            if ((GTot1.node[q]).get('state')).find('i')==0:
                GTot1.node[q]['color']='grey'
            if ((GTot1.node[q]).get('state')).find('h')==0:
                GTot1.node[q]['color']='red' 


#determining color and size of edges
    for q in GTot1.edges():
        GTot1[q[0]][q[1]]['color']='orange'
        GTot1[q[0]][q[1]]['size']=0.5

    
    for q in GTot1.edges():
        if (((GTot1.node[q[0]]).get('state')).find('i')==0): 
            if q[1].find('o1')==0:
                ch=0
                for k in GTot1.neighbors(q[1]):
                    if ((GTot1.node[k]).get('state')).find('h')==0:
                        ch=1
                if ch==0:
                    GTot1[k][q[1]]['color']='grey'
                    GTot1[k][q[1]]['size']=4
                else:
                    GTot1[k][q[1]]['color']='red'
                    GTot1[k][q[1]]['size']=4
            else:
                if q[1].find('o')==0:
                    for k in GTot1.neighbors(q[1]):
                        GTot1[k][q[1]]['color']='grey'
                        GTot1[k][q[1]]['size']=4
                    GTot1[q[0]][q[1]]['color']='grey'
                    GTot1[q[0]][q[1]]['size']=4
                else:
                    GTot1[q[0]][q[1]]['color']='grey'
                    GTot1[q[0]][q[1]]['size']=4
        if (((GTot1.node[q[1]]).get('state')).find('i')==0):
            if q[0].find('o1')==0:
                ch=0
                for k in GTot1.neighbors(q[0]):
                    if ((GTot1.node[k]).get('state')).find('h')==0:
                        ch=1
                if ch==0:
                    GTot1[k][q[0]]['color']='grey'
                    GTot1[k][q[0]]['size']=4
                else:
                    GTot1[k][q[0]]['color']='red'
                    GTot1[k][q[0]]['size']=4
            else:
                if q[0].find('o')==0:
                    for k in GTot1.neighbors(q[0]):
                        GTot1[k][q[0]]['color']='grey'
                        GTot1[k][q[0]]['size']=4
                    GTot1[q[0]][q[1]]['color']='grey'
                    GTot1[q[0]][q[1]]['size']=4
                else:
                    GTot1[q[0]][q[1]]['color']='grey'
                    GTot1[q[0]][q[1]]['size']=4
    
    for q in GTot1.nodes():
        if ((GTot1.node[q]).get('state')).find('h')==0:
            GTot1[q]['o1']['color']='red'
            GTot1[q]['o1']['size']=4
            for k in GTot1.neighbors('o1'):
                GTot1[k]['o1']['color']='red'
                GTot1[k]['o1']['size']=4
            
    for q in GTot1.edges():
        if (q[0].find('o')==0): 
            if (q[1].find('F')==0) and (((GTot1.node[q[1]]).get('state')).find('i')!=0):
                ch=0
                for k in GTot1.neighbors(q[0]):
                    if (((GTot1.node[k]).get('state')).find('h')==0) or (((GTot1.node[k]).get('state')).find('i')==0):
                        ch=1
                if ch==0:
                    GTot1[q[0]][q[1]]['color']='orange'
                    GTot1[q[0]][q[1]]['size']=0.5
        if (q[1].find('o')==0): 
            if (q[0].find('F')==0) and (((GTot1.node[q[0]]).get('state')).find('i')!=0):
                ch=0
                for k in GTot1.neighbors(q[1]):
                    if (((GTot1.node[k]).get('state')).find('h')==0) or (((GTot1.node[k]).get('state')).find('i')==0):
                        ch=1
                if ch==0:
                    GTot1[q[0]][q[1]]['color']='orange'
                    GTot1[q[0]][q[1]]['size']=0.5
                
    for q in GTot1.edges():
        if (((GTot1.node[q[0]]).get('state')).find('r')==0) or (((GTot1.node[q[1]]).get('state')).find('r')==0):
            GTot1[q[0]][q[1]]['color']='green'
            GTot1[q[0]][q[1]]['size']=0.5
     
    return GTot1







def next_step(dayduration,t,GTot1,Gpop,Go,Gr1,Gr2,Gr3,Gr4,Gr5,Gromerged,GSW,nlistmerged,quarantine,bt,alphatemp,betatemp,mobility,pmobility,mobfreq,N1,N2,N3,N4,N5,posocdist,po,p,psocdist,hospitalized1,infected1,recovered1,susceptible1,other1,betafree):
        #resetting times of interactions for the new step
    if t%(dayduration)==0:
        for q in GTot1.edges():
            time1=random.random()
            GTot1[q[0]][q[1]]['time']=time1
        
        GTot2=GTot1.copy()
        GTot2.remove_edges_from(Gpop.edges())
        for q in GTot2.edges():
            if ((q[0].find('F')==0) or (q[0].find('o1')==0)) and ((q[1].find('F')==0) or (q[1].find('o1')==0)):
                GTot1[q[0]][q[1]]['time']=1
        
        betafree=random.gammavariate(alphatemp[bt+0],betatemp[bt+0])
        for q in GTot1.nodes():
            if q.find('o')==0:
                GTot1.node[q]['social temp']=random.gammavariate(alphatemp[bt+1],betatemp[bt+1])
        GTot1.node['o1']['social temp']=random.gammavariate(alphatemp[bt+2],betatemp[bt+2])

    
#updating viral load    
    for q in hospitalized1:
        viralload1=random.random()
        GTot1.node[q]['viral load']=1/2+viralload1/2
    
    for q in infected1:
        viralload1=random.random()
        GTot1.node[q]['viral load']=1/4+viralload1/4
    
    for q in recovered1:
        viralload1=random.random()
        GTot1.node[q]['viral load']=viralload1/4
    
    for q in susceptible1:
        GTot1.node[q]['viral load']=0
    
    for q in other1:
        viralload1=random.random()
        GTot1.node[q]['viral load']=1/2+viralload1/2
    
    if mobility==1:
        if t%(mobfreq)==0:
            GTot3=GTot1.copy()
            for qi in range(N1):
                GTot3.remove_edges_from(Gr1[qi].nodes())
            
            for qi in range(N2):
                GTot3.remove_edges_from(Gr2[qi].edges())
                
            for qi in range(N3):
                GTot3.remove_edges_from(Gr3[qi].edges())
                
            for qi in range(N4):
                GTot3.remove_edges_from(Gr4[qi].edges())
                
            for qi in range(N5):
                GTot3.remove_edges_from(Gr5[qi].edges())
            
            for j, q in list(GTot3.edges()):
                if random.random() < pmobility:
                    if (((GTot1.node[j]).get('state')).find('h')!=0) and (((GTot1.node[q]).get('state')).find('h')!=0) and (GTot1.node[j]['detected'] != 'yes') and (GTot1.node[q]['detected'] != 'yes') and (GTot1.has_edge(j,q)==True):
                        GTot1.remove_edge(j,q)
                        if (Gpop.has_edge(j,q)==True):
                            Gpop.remove_edge(j,q)
                            
                        elif (Go.has_edge(j,q)==True):
                            Go.remove_edge(j,q)
                            
                        if j.find('o')!=0:
                            if quarantine==0:
                                ps=p
                                pso=po
                            else:
                                ps=psocdist
                                pso=posocdist
                            if random.random()<(pso/(ps+pso)):
                                qq=random.randint(0,len(Gromerged)-1)
                                w1=Gromerged[qq]
                            else:    
                                for q1 in range(N1-1):
                                    if j in nlistmerged[q1].nodes():
                                        s2=q1
                                s1=random.randint(0,N1-1)
                                while (GSW.has_edge(s1,s2)==False): #select two houses at random that are connected in GSW
                                    s1=random.randint(0,N1)
                                ss1=random.randint(0,len(nlistmerged[s1].nodes())-1)
                                w1=nlistmerged[s1].nodes()[ss1]
                                while (GTot3.has_edge(j,w1)==True) or (j in nlistmerged[s1].nodes()) or (((GTot1.node[w1]).get('state')).find('h')==0) or (GTot1.node[w1]['detected'] == 'yes'):
                                    s1=random.randint(0,N1-1)
                                    while (GSW.has_edge(s1,s2)==False): #select two houses at random that are connected in GSW
                                        s1=random.randint(0,N1-1)
                                    ss1=random.randint(0,len(nlistmerged[s1].nodes())-1)
                                    w1=nlistmerged[s1].nodes()[ss1]
                        else:
                            w2=random.randint(0,len(Gpop.nodes())-1)
                            w1=list(Gpop.nodes())[w2]
                        if (j.find('o')==0) or (w1.find('o')==0): 
                            Go.add_edge(j, w1)
                            Go[j][w1]['time']=random.random()
                            
                        else:
                            Gpop.add_edge(j, w1)
                            Gpop[j][w1]['time']=random.random()
                            
                        GTot1.add_edge(j, w1)
                        GTot1[j][w1]['time']=random.random()
 
    return GTot1,Go,Gpop,betafree
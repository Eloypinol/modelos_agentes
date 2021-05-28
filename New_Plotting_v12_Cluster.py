# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:38:31 2020

@author: Chris
"""

import numpy as np

#import random

import matplotlib.pyplot as plt

import networkx as nx

import math

#import itertools

#import time

#import scipy.special as sps

#from scipy.stats import gamma

#import pandas as pd

#from math import log10, floor
#parameters


outfilename=r"pruebas/prueba1"
outputfile = "pruebas/prueba1/"

tsocdistv1=[1500]
#tsocdistv1=[6250]
#tsocdistv1=[1500,6250]

tsocdistliftv1=[11250]
#tsocdistliftv1=[6250]
#tsocdistliftv1=[16250]
#tsocdistliftv1=[11250,11250]

mobilityv1=[0]
#mobilityv1=[1]
#mobilityv1=[0,0]

resumingv1=[0]
#resumingv1=[0,0]

runnv1=[0]
#runnv1=[0,0]

sume=375895.8559222643
sume=375895.87752226426
sume=375895.87752226426
#sume=375895.8559222643
sumev=[377490.7043821206]

#sumev=[377490.30438212055]
sumev=[377491.10438212054]
sumev=[377491.2643821205]
sumev=[317490.7043821205]
sumev=[377490.7043821206]
sumev=[437490.7043821206]
sumev=[377490.7043821206]
sumev=[375896.8559222643]
sumev=[377490.1443821205]
sumev=[377497.5043821205]
sumev=[377494.50438212056]
sumev=[377491.3343821206]

sumev = [375895.8559222643]

#sumev=[377482.9043821205]
#sumev=[377481.7043821206]

#sumev=[429740.7043821206]

#sumev=[377490.7043821206,429740.7043821206]

compvecMAXHOSP=[]
compvecMAXDIE=[]
compvecMAXINF=[]
compvecMAXINFDAILY=[]
compvecMAXTIME=[]

compvecMAXHOSPvar=[]
compvecMAXDIEvar=[]
compvecMAXINFvar=[]
compvecMAXINFDAILYvar=[]
compvecMAXTIMEvar=[]

Nv=[]
nov=[]
n2v=[]
n3v=[]
n4v=[]
n5v=[]
pininfv=[]
daydurationv=[]
prewirev=[]
pv=[]
tsocdistv=[]
tsocdistliftv=[]
psocdistv=[]
pov=[]
posocdistv=[]
R01v=[]
pdv=[]
quarantinev=[]
pkv=[]
pmobility1v=[]
pmobilitysocdistv=[]
pmobilityafterv=[]
mobilityv=[]
mobfreqv=[]
tautempfreev=[]
tautempplacev=[]
tautemphospv=[]
tautempfreequav=[]
tautempplacequav=[]
tautemphospquav=[]
tautempfreeafterquav=[]
tautempplaceafterquav=[]
tautemphospafterquav=[]
tausick1v=[]
sigmasickv=[]
taurecovered1v=[]
sigmarecoveredv=[]
tauinfected1v=[]
sigmainfectedv=[]
taurec1v=[]
sigmarecv=[]
tauhosp1v=[]
sigmahospv=[]



for k1 in range(len(sumev)):
    compvecMAXHOSP1=[]
    compvecMAXDIE1=[]
    compvecMAXINF1=[]
    compvecMAXINFDAILY1=[]
    compvecMAXTIME1=[]
    for k2 in range(runnv1[k1]+1):

        dat=np.loadtxt(outfilename+"/Sim_sume_"+str(sumev[k1])+"_tsocdist_"+str(tsocdistv1[k1])+"_tsocdistlift_"+str(tsocdistliftv1[k1])+"_mobility_"+str(mobilityv1[k1])+"_resuming_"+str(resumingv1[k1])+"_runn_"+str(k2)+".csv",dtype=str,delimiter=',',comments='#')
        datR0=np.loadtxt(outfilename+"/R0_sume_"+str(sumev[k1])+"_tsocdist_"+str(tsocdistv1[k1])+"_tsocdistlift_"+str(tsocdistliftv1[k1])+"_mobility_"+str(mobilityv1[k1])+"_resuming_"+str(resumingv1[k1])+"_runn_"+str(k2)+".csv",dtype=str,delimiter=',',comments='#')
        
        para=np.loadtxt(outfilename+"/Par_sume_"+str(sumev[k1])+"_tsocdist_"+str(tsocdistv1[k1])+"_tsocdistlift_"+str(tsocdistliftv1[k1])+"_mobility_"+str(mobilityv1[k1])+"_resuming_"+str(resumingv1[k1])+"_runn_"+str(k2)+".csv",dtype=str,delimiter=',',comments='#')
        
                        
                        
                        
                        
        N=int(para[0][1]) #total population
    
    
        numberofplaces=int(para[1][1]) #total number of places (hospital,transport,work)
    
            
        print('Total population: ',N-numberofplaces)
        print('Number of places: ',numberofplaces)
        n2=float(para[2][1]) #percentage of 2 person families
    
        
        n3=float(para[3][1]) #percentage of 3 person families
    
        
        n4=float(para[4][1]) #percentage of 4 person families
    
        
        n5=float(para[5][1]) #percentage of 5 person families
    
        #rest is single person groups
        
        pininf=float(para[6][1]) #percentage of initially infected people
        print('Number of initially infected people: ',math.ceil(pininf*N))
    
        
        dayduration=int(para[7][1])#math.ceil(0.1*N) #how many steps in a day
        print('duration of a day: ',dayduration)
        
        prewire=float(para[8][1]) #probbility to rewire an edge for the small-world graph of the connections between people of different groups
        print('Probability to rewire in small-world: ',prewire)
    
        
        p=float(para[9][1])# probability to accept an edge of the small world-graph 
        print('Small-world parameter: ',p)
    
        
        tsocdist=int(float(para[10][1]))#5000*dayduration #time at which social distancing measures are implemented
        print('Day social distancing will be imposed: ',tsocdist)
    
        
        tsocdistlift=int(float(para[11][1]))
        print('Day social distancing will be lifted: ',tsocdistlift)
    
        
        psocdist=float(para[12][1]) #percentage of social connections removed
        print('Percentage of social connections removed: ',psocdist)
    
        
        po=float(para[13][1])# probability of a connection between a citizen and a place 
        print('Probability of a connection between a citizen and a place: ',po)
    
        
        posocdist=float(para[14][1]) # probability to remove these connections between people and places after quarantine measures
        print('Probability to remove these connections between people and places after quarantine measures: ',posocdist)
    
        
        #ph=0.005 #probability of getting hospitalized at each time step after being infected
        #pr=TIME FROM ADMISSION TO DISCHARGE 0.05 #probability to recover at each time step
        R01=int(para[15][1]) #number of people a person can infect
    
        
        
        pd=float(para[16][1]) #probability of an infected person to be detected
        print('Probability of an infected person to be detected: ',pd)
    
        
        quarantine=int(para[17][1]) #whether we start with a quarantine setup or not (0 is deactivated, 1 is activated)
    
        
        
        #beta=10**-6 #beta of the ising model
        
        
        N2=math.ceil(N*n2)//2;
        print('Number of people in 2-persons houses: ',N2)
        N3=(math.ceil(N*n3))//3;
        print('Number of people in 3-persons houses: ',N3)
        N4=(math.ceil(N*n4))//4;
        print('Number of people in 4-persons houses: ',N4)
        N5=math.ceil(N*n5)//5;
        print('Number of people in 5-persons houses: ',N5)
        No=numberofplaces;
        N1=N-5*N5-2*N2-3*N3-4*N4-No;
        print('Number of people in 1-persons houses: ',N1)
        
        NTotH=N1+N2+N3+N4+N5 #total number of houses
        print('total number of houses: ',NTotH)
        pk=float(para[18][1])
    
        
        knearestneighbors=math.ceil(pk*NTotH)
        print('knearestneighbors in small-world: ',knearestneighbors)
        
        pmobility1=float(para[19][1])
        print('Probability to switch neighbor if mobility is allowed: ',pmobility1)
    
        
        pmobilitysocdist=float(para[20][1])
        print('Probability to switch neighbor if mobility is allowed under social distancing measures: ',pmobilitysocdist)
    
        
        pmobilityafter=float(para[21][1])
        print('Probability to switch neighbor if mobility is allowed after social distancing measures: ',pmobilityafter)
    
    
        mobility=int(para[22][1])
        print('Is mobility allowed? ',mobility)
    
        
        
        mobfreq=float(para[23][1])
        print('Frequency of switching neighbors if mobility is allowed: ',mobfreq)
    
        
        
        tautempfree=float(para[24][1])
        print('Average contagious temperature among people: ',tautempfree)
    
        
        tautempplace=float(para[25][1])
        print('Average contagious temperature in places: ',tautempplace)
    
        
        tautemphosp=float(para[26][1])
        print('Average contagious temperature in hospital: ',tautemphosp)
    
        
        tautempfreequa=float(para[27][1])
        print('Average contagious temperature among people in quarantine conditions: ',tautempfreequa)
    
    
        tautempplacequa=float(para[28][1])
        print('Average contagious temperature in places in quarantine conditions: ',tautempplacequa)
    
        
        tautemphospqua=float(para[29][1])
        print('Average contagious temperature in hospital in quarantine conditions: ',tautemphospqua)
        
        
        tautempfreeafterqua=float(para[30][1])
        print('Average contagious temperature among people after quarantine conditions: ',tautempfreeafterqua)
    
        tautempplaceafterqua=float(para[31][1])
        print('Average contagious temperature in places after quarantine conditions: ',tautempplaceafterqua)
    
        tautemphospafterqua=float(para[32][1])
        print('Average contagious temperature in hospital after quarantine conditions: ',tautemphospafterqua)
        
        
        
        tausick1=float(para[33][1])#*dayduration #average time a person stays sick in the hospital (normal dist)
        print('Average time a person stays sick in the hospital before dying: ',tausick1)
        sigmasick=float(para[34][1]) #standard deviation of sick time
        print('Variance of time a person stays sick in the hospital before dying: ',sigmasick)

        
        
        taurecovered1=float(para[35][1])#*dayduration #average time it takes for a person to recover (normal dist)
        print('Average time it takes for a recovered person to become susceptible: ',taurecovered1)
        sigmarecovered=float(para[36][1]) #standard deviation of recovery time
        print('Variance of time it takes for a recovered person to become susceptible: ',sigmarecovered)

        
        tauinfected1=float(para[37][1])#*dayduration #average time a person stays infected but without symptoms (normal dist) (just carrying the virus)
        print('Average time a person stays infected before recovering: ',tauinfected1)
        sigmainfected=float(para[38][1]) #standard deviation of time being infected
        print('Variance of time a person stays infected before recovering: ',sigmainfected)

        
        
        taurec1=float(para[39][1])#*dayduration #average time it takes for a person to recover (normal dist)
        print('Average time for a person to recover from hospitl: ',taurec1)
        sigmarec=float(para[40][1]) #standard deviation of recovery time
        print('Variance of time for a person to recover from hospitl: ',sigmarec)

        
        
        tauhosp1=float(para[41][1])#*dayduration #average time it takes for a person to recover (normal dist)
        print('Average time it takes for infected person to go to hospital: ',tauhosp1)
        sigmahosp=float(para[42][1]) #standard deviation of recovery time
        print('Variance of time it takes for infected person to go to hospital: ',sigmahosp)

    
        #runn=runnv[k1]
                    
                        
                        
                        
        Vecs=[Nv,nov,n2v,n3v,n4v,n5v,pininfv,daydurationv,prewirev,pv,tsocdistv,tsocdistliftv,psocdistv,pov,posocdistv,R01v,pdv,quarantinev,pkv,pmobility1v,pmobilitysocdistv,pmobilityafterv,mobfreqv,tautempfreev,tautempplacev,tautemphospv,tautempfreequav,tautempplacequav,tautemphospquav,tautempfreeafterquav,tautempplaceafterquav,tautemphospafterquav,tausick1v,sigmasickv,taurecovered1v,sigmarecoveredv,tauinfected1v,sigmainfectedv,tauhosp1v,sigmahospv,taurec1v,sigmarecv,mobilityv]    
        VecCur=[N,numberofplaces,n2,n3,n4,n5,pininf,dayduration,prewire,p,tsocdist,tsocdistlift,psocdist,po,posocdist,R01,pd,quarantine,pk,pmobility1,pmobilitysocdist,pmobilityafter,mobfreq,tautempfree,tautempplace,tautemphosp,tautempfreequa,tautempplacequa,tautemphospqua,tautempfreeafterqua,tautempplaceafterqua,tautemphospafterqua,tausick1,sigmasick,taurecovered1,sigmarecovered,tauinfected1,sigmainfected,tauhosp1,sigmahosp,taurec1,sigmarec,mobility]    
        
        for ss1 in range(len(Vecs)):
            if VecCur[ss1] not in Vecs[ss1]:
                Vecs[ss1].append(VecCur[ss1])
        
                        
                         
        
        totinitinf=math.ceil(pininf*N)
        
        tnewinf=[0]
        tnewinfdet=[]
        tnewdead=[]
        tnewrecid=[]
        tnewreci=[]
        tnewrech=[]
        tnewhospd=[]
        tnewhosp=[]
        
        #print(dat[len(dat)-1][0])
        stoptime=int(float((dat[len(dat)-1][0])))
        print('duration: ',stoptime)
        
        
        newinf=[]
        newinfdet=[0]
        newdead=[0]
        newhosp=[0]
        newhospd=[0]
        newreci=[0]
        newrecid=[0]
        newrech=[0]
        
        dailinf=[]
        dailinfdet=[0]
        dailhosp=[0]
        dailhospd=[0]
        
        
        totinf=[]
        totinfdet=[0]
        totdead=[0]
        tothospd=[0]
        tothosp=[0]
        totrech=[0]
        totrecid=[0]
        totreci=[0]
        
        
            
        lista=[0]
        for j in range(10,len(dat)):
            if dat[j][0]!='':
                lista.append(float(dat[j][0]))
#        
#        print('lista: ',lista[10])
        #    else:
        #        lista.append(1501)
        
        #print(lista)
        
        newinf.append(totinitinf)
        totinf.append(totinitinf)
        dailinf.append(totinitinf)
        
        susu=0
        print('len: ',len(dat))
        for i in range(10,len(dat)):
            changed=0
            if ("infected but not detected" in dat[i][1]) or ("infected and detected" in dat[i][1]):
                #tnewinf.append(int(float(dat[i][0])))
                newinf.append(1)
                totinf.append(totinf[len(totinf)-1]+1)
                dailinf.append(dailinf[len(dailinf)-1]+1)
                changed=1
                #susu+=1
            else:
                newinf.append(0)
                totinf.append(totinf[len(totinf)-1])
                 
            if "infected and detected" in dat[i][1]:
                #tnewinfdet.append(int(float(dat[i][0])))
                newinfdet.append(1)
                totinfdet.append(totinfdet[len(totinfdet)-1]+1)
                changed=2
                dailinfdet.append(dailinfdet[len(dailinfdet)-1]+1)
                #dailinf.append(dailinf[len(dailinf)-1]+1)
            else:
                newinfdet.append(0)
                totinfdet.append(totinfdet[len(totinfdet)-1])
                
                 
            if (("recovered from infected detected" in dat[i][1]) or ("recovered from infected not detected" in dat[i][1])) and (changed!=1):
                #tnewreci.append(int(float(dat[i][0])))
                newreci.append(1)
                totreci.append(totreci[len(totreci)-1]+1)
                dailinf.append(dailinf[len(dailinf)-1]-1)
                #dailinf[len(dailinf)-1]-=1
                changed=3
                #susu+=1
            else:
                newreci.append(0)
                #totreci.append(totreci[len(totreci)-1])
            
            if ("recovered from infected detected" in dat[i][1]):
                #tnewrecid.append(int(float(dat[i][0])))
                newrecid.append(1)
                totrecid.append(totrecid[len(totrecid)-1]+1)
                #dailinfdet[len(dailinfdet)]-=1
                dailinfdet.append(dailinfdet[len(dailinfdet)-1]-1)
                changed=4
            else:
                newrecid.append(0)
                totrecid.append(totrecid[len(totrecid)-1])
            
            if (("hospitalized no quarantine" in dat[i][1]) or ("hospitalized from detected" in dat[i][1])) and (changed!=1) and (changed!=3):
                #tnewhosp.append(int(float(dat[i][0])))
                newhosp.append(1)
                tothosp.append(tothosp[len(tothosp)-1]+1)
                dailhosp.append(dailhosp[len(dailhosp)-1]+1)
                dailinf.append(dailinf[len(dailinf)-1]-1)
                changed=5
                #susu+=1
                #print('hola')
            else:
                newhosp.append(0)
                tothosp.append(tothosp[len(tothosp)-1])
                #dailinf.append(dailinf[len(dailinf)-1])
                #dailhosp.append(dailhosp[len(dailhosp)-1])
            if "hospitalized from detected" in dat[i][1]:
                #tnewhospd.append(int(float(dat[i][0])))
                newhospd.append(1)
                tothospd.append(tothospd[len(tothospd)-1]+1)
                #tothosp.append(tothosp[len(tothosp)-1]+1)
                #dailhosp.append(dailhosp[len(dailhosp)-1]+1)
                #dailinf[len(dailinf)]-=1
                dailinfdet.append(dailinfdet[len(dailinfdet)-1]-1)
                changed=6
            else:
                newhospd.append(0)
                tothospd.append(tothospd[len(tothospd)-1])
                #dailhosp.append(dailhosp[len(dailhosp)-1])
            if "died" in dat[i][1]:
                #tnewdead.append(int(float(dat[i][0])))
                newdead.append(1)
                totdead.append(totdead[len(totdead)-1]+1)
                dailhosp.append(dailhosp[len(dailhosp)-1]-1)
                #print('hola')
                changed=7
                #susu+=1
                #dailhosp[len(dailhosp)-1]-=1
            else:
                newdead.append(0)
                totdead.append(totdead[len(totdead)-1])
            if "recovered from hospital" in dat[i][1]:
                #tnewrech.append(int(float(dat[i][0])))
                newrech.append(1)
                totreci.append(totreci[len(totreci)-1]+1)
                totrech.append(totrech[len(totrech)-1]+1)
                dailhosp.append(dailhosp[len(dailhosp)-1]-1)
                #print('hola')
                changed=8
                #susu+=1
                #dailhosp[len(dailhosp)-1]-=1
            else:
                newrech.append(0)
                totrech.append(totrech[len(totrech)-1])
                
            if (changed!=1) and (changed!=3) and (changed!=5) and (changed!=6) and (changed!=2) and (changed!=4):
                #print('hola')
                susu+=1
                dailinf.append(dailinf[len(dailinf)-1])
            if (changed!=5) and (changed!=7) and (changed!=8) and (changed!=6):
                dailhosp.append(dailhosp[len(dailhosp)-1])
                #susu+=1
            if (changed!=2) and (changed!=4)and (changed!=6):
                dailinfdet.append(dailinfdet[len(dailinfdet)-1])
            if (changed!=3) and (changed!=8) and (changed!=4):
                #print('hola',changed)
                totreci.append(totreci[len(totreci)-1])
                
#            else:
#                newinf.append(0)
#                totinf.append(totinf[len(totinf)-1])
#                dailinf.append(dailinf[len(dailinf)-1])
#                newinfdet.append(0)
#                totinfdet.append(totinfdet[len(totinfdet)-1])
#                dailinfdet.append(dailinfdet[len(dailinfdet)-1])
#                newdead.append(0)
#                totdead.append(totdead[len(totdead)-1])    
#                newreci.append(0)
#                totreci.append(totreci[len(totreci)-1])
#                newrecid.append(0)
#                totrecid.append(totrecid[len(totrecid)-1])
#                newhosp.append(0)
#                tothosp.append(tothosp[len(tothosp)-1])
#                dailhosp.append(dailhosp[len(dailhosp)-1])
#                newhospd.append(0)
#                tothospd.append(tothospd[len(tothospd)-1])
#                newrech.append(0)
#                totrech.append(totrech[len(totrech)-1])
#                #dailhospd.append(dailhospd[len(dailhospd)-1])
                
                
        print('newinf: ',sum(newinf))
                    
        print('newhosp:',sum(newhosp)) 
        print(dailinf[1])           
        print('newdead:',sum(newdead))     
        
        newchanges=[newinf,newinfdet,newdead,newreci,newhosp,newrecid,newrech,totinf,totinfdet,totdead,totreci,tothosp,totrecid,totrech,dailinf,dailinfdet,dailhosp]
        newchanges=[newinf,newinfdet,newdead,newreci,newhosp,totinf,totinfdet,totdead,totreci,tothosp,dailinf,dailinfdet,dailhosp]

        #print('dailhosp:',dailhosp)
        
        
        #print(len(newchanges[12])) #dailhosp
        #print(sum(newchanges[12]))
        #print(len(newchanges[10]))#dailinf
        #print(sum(newchanges[10]))
        #print(len(newchanges[8]))#totreci
        #print(sum(newchanges[8]))
#        print(len(newchanges[2]))#totdead
#        print(sum(newchanges[2]))
        print(len(lista))
        print(susu)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#        newvecsave=[]
#        for i in range(5):
#            yt=0
#            ytd=1
#            if i==0:
#                newvec=[0,totinitinf]
#            else:
#                newvec=[0, 0]
#            newvec1=0
#            while yt<len(newchanges[i]):
#                if (yt//dayduration>(ytd-1)):
#                    #j+=1
#                    ytd+=1
#                    #day1=day[j]
#                    newvec1=0
#                    newvec.append(newvec1)
#                else:
#                    newvec1=newvec1+newchanges[i][yt]
#                newvec[ytd]=newvec1
#                yt+=1
#            newvecsave.append(newvec)
#            #plotnew[i],=plt.plot(timeaxisD,newvec)
#        print(newvecsave[0])
#        print(sum(newvecsave[0]))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        print(lista[1])
        print(newchanges[0][1])
        
        
        
        
        
        
        
        
        
        # PLOT OF NEW CASES
        
        
        
        
        
        totdays=stoptime//dayduration
        print('totdays: ',totdays)
        #timeaxisD=int(np.divide(lista,dayduration))
        timeaxisD=range(totdays+1)
        
        plotnew=[]
        plotnew1=[]
        plotnew2=[]
        
        for i in range(len(newchanges)+1):
            plotnew.append(0)
            plotnew1.append(0)
            plotnew2.append(0)
        
        newvecsave=[]
        totvecsave=[]
        dailvecsave=[]
        for i in range(len(newchanges)):
            timeblock=0
            timeaxis1=[0]
            newvec1=0
#            if i==0:
#                newvec=[totinitinf]
#            else:
#                newvec=[0]
            if i<5:
                newvec=[newchanges[i][0]]
            if (i>=5) and (i<10):
                totvec=[newchanges[i][0]]
            if i>=10:
                dailvec=[newchanges[i][0]]
            for j in range(1,len(lista)):
                #print(lista[10])
                if (lista[j]<(timeblock+1)*dayduration) and (lista[j]>=timeblock*dayduration):
                    if i<5:
                        newvec1+=newchanges[i][j]
                    if j==len(lista)-1:
                        newvec.append(newvec1)
                        if (i>=5) and (i<10):
                            totvec.append(newchanges[i][j])
                        if i>=10:
                            dailvec.append(newchanges[i][j])
                    #print('hola')
                else:
                    timeblock=lista[j]//dayduration
                    timeaxis1.append(timeblock)
                    if i<5:
                        newvec.append(newvec1)
                        newvec1=newchanges[i][j]
                    if (i>=5) and (i<10):
                        totvec.append(newchanges[i][j-1])
                    if i>=10:
                        dailvec.append(newchanges[i][j-1])
            if i<5:        
                newvecsave.append(newvec)
            if (i>=5) and (i<10):
                totvecsave.append(totvec)
            if i>=10:
                dailvecsave.append(dailvec)

        
        
        #print('lista: ',lista)
        #print(totvecsave[0])
        
        timeaxis1=timeaxis1[0:len(timeaxis1)-1]
        timeaxis1.append(timeaxis1[len(timeaxis1)-1])
        print('timeaxis1: ',timeaxis1)
        socialdista=[0 for ui in range(len(dailvecsave[2]))]
        
        #socialdista[6]=300
        #socialdista[45]=300
        
        fig0 = plt.figure(figsize=(12.5,7.5))
        axe0=fig0.gca()
        plt.title('New cases VS time')
        for ss in range(5):
            timeaxisD=range(len(newvecsave[ss]))
            plotnew[ss],=plt.plot(timeaxis1,newvecsave[ss])
            #if ss==4:
                #plotnew[5],=plt.plot(timeaxisD,socialdista)
        #plotnew[1],=plt.plot(timeaxisD,newvecsave[1])
        plt.legend([plotnew[0],plotnew[1],plotnew[2],plotnew[3],plotnew[4]], ['new infected','new infected detected','new dead','new recovered','new hospitalized'],fontsize=20)
        axe0.set_xlabel('time',fontsize=30)
        axe0.set_ylabel('population',fontsize=30)
        plt.savefig(outputfile+"New_VS_time_"+str(sumev[k1]) + '.png')
            
            
        fig1 = plt.figure(figsize=(12.5,7.5))
        axe1=fig1.gca()
        plt.title('Total cases VS time')
        for ss in range(5):
            timeaxisD=range(len(totvecsave[ss]))
            plotnew1[5+ss],=plt.plot(timeaxis1,totvecsave[ss])
            
        #plotnew[1],=plt.plot(timeaxisD,newvecsave[1])
        plt.legend([plotnew1[5],plotnew1[6],plotnew1[7],plotnew1[8],plotnew1[9]], ['total infected','total infected detected','total dead','total recovered','total hospitalized'],fontsize=20)
        axe1.set_xlabel('time',fontsize=30)
        axe1.set_ylabel('population',fontsize=30)
        plt.savefig(outputfile+"Total_VS_time_"+str(sumev[k1]) + '.png')   
            
            
        fig2 = plt.figure(figsize=(12.5,7.5))
        axe2=fig2.gca()
        plt.title('Daily cases VS time')
        for ss in range(3):
            timeaxisD=range(len(dailvecsave[ss]))
            plotnew2[10+ss],=plt.plot(timeaxis1,dailvecsave[ss])
            #if ss==2:
                #plotnew2[13],=plt.plot(timeaxisD,socialdista)
        plt.axvline(x=6,c='r',ls='--')
        plt.axvline(x=45,c='r',ls='--')
        #plotnew[1],=plt.plot(timeaxisD,newvecsave[1])
        plt.legend([plotnew2[10],plotnew2[11],plotnew2[12]], ['daily infected','daily infected detected','daily hospitalized'],fontsize=20)
        axe2.set_ylabel('population',fontsize=30)
        axe2.set_xlabel('time',fontsize=30)
        plt.savefig(outputfile+"Daily_VS_time_"+str(sumev[k1]) + '.png')    
            
            
            
            
            
            
            
          #PLOT OF R0 METHOD 1
        
        
        
        
        
        fig3 = plt.figure(figsize=(12.5,7.5))
        axe3=fig3.gca()
        #print(len(newvecsave[0]))
        #print(len(dailvecsave[0]))
        rat=[]
        for q in range(len(dailvecsave[0])-1):
            if (dailvecsave[0][q]+dailvecsave[2][q])>0:
                rat.append(newvecsave[0][q+1]/(dailvecsave[0][q]+dailvecsave[2][q]))
            
        plotnew3,=plt.plot(range(len(rat)),rat)
        #plotnew[1],=plt.plot(timeaxisD,newvecsave[1])
        plt.legend([plotnew3], ['R0'],fontsize=20)
        axe3.set_xlabel('time',fontsize=30)
        plt.savefig(outputfile+"R0_VS_time_1_"+str(sumev[k1]) + '.png')
        #plt.savefig("New_VS_time_"+str(sume) + '.png')
        
        #print(dailinf)
        
        
        
        
        
#        # PLOT OF R0 METHOD 2
#        
#        
#        
#        
#        GR0=nx.Graph()
#        listaR0=[]
#        for j in range(len(datR0)):
#            listaR0.append(int(float(datR0[j][0])))
#            GR0.add_node(datR0[j][1])
#            GR0.node[datR0[j][1]]['infect']=0
#        
#        #print(GR0.nodes(data=True))
#        #print(listaR0)
#        cc=0
#        infe=[[],[]]
#        for u in range(listaR0[len(listaR0)-1]//dayduration):
#            #print(dayduration*u)
#            #print(listaR0[cc])
#            while (listaR0[cc]<dayduration*u) and (listaR0[cc]>=dayduration*(u-1)):
#                #print(cc)
#                GR0.node[datR0[cc][1]]['infect']+=1
#                cc+=1
#            infe[0].append(u)
#            suma1=0
#            infe1=[v for v in GR0.node if GR0.node[v]['infect']>0]
#            #print(infe1)
#            for rr in infe1:
#                #print(GR0.node[rr]['infect'])
#                suma1=suma1+GR0.node[rr]['infect']
#            if len(infe1)==0:
#                infe[1].append(0)
#            else:
#                infe[1].append(suma1/len(infe1))
#        
#        #print(infe)
#        #print(GR0.nodes(data=True))
#        tim=infe[0]
#        infec=infe[1]
#        
#        
#    
#        
#        
#        
#        fig4 = plt.figure(figsize=(12.5,7.5))
#        axe4=fig4.gca()
#        
#        plotnew4,=plt.plot(tim,infec)
#        #plotnew[1],=plt.plot(timeaxisD,newvecsave[1])
#        plt.legend([plotnew4], ['R0'],fontsize=20)
#        axe4.set_xlabel('time',fontsize=30)
#        plt.savefig("R0_VS_time_2_"+str(sumev[k1]) + '.png')   
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
#        print('newinf: ',sum(newvecsave[2]))
#        print('newinf: ',newvecsave[2])
#        print('newinf: ',len(newvecsave[2]))
#        
#        for ui in range(len(newvecsave[2])):
#            if newvecsave[2][ui]!=0:
#                print(ui*250)
#        
#
#
#        
##        #timeaxisD=range(totdays+1)
##        newvecsave=[]
##        for i in range(5):
##            yt=0
##            ytd=1
##            if i==0:
##                newvec=[0,totinitinf]
##            else:
##                newvec=[0, 0]
##            newvec1=0
##            while yt<len(newchanges[i]):
##                if (yt//dayduration>(ytd-1)):
##                    #j+=1
##                    ytd+=1
##                    #day1=day[j]
##                    newvec1=0
##                    newvec.append(newvec1)
##                else:
##                    newvec1=newvec1+newchanges[i][yt]
##                newvec[ytd]=newvec1
##                yt+=1
##            newvecsave.append(newvec)
##            #plotnew[i],=plt.plot(timeaxisD,newvec)
#        
#        #print(newvecsave[0])
#        
#        fig0 = plt.figure(figsize=(12.5,7.5))
#        axe0=fig0.gca()
#        for ss in range(5):
#            timeaxisD=range(len(newvecsave[ss]))
#            plotnew[ss],=plt.plot(timeaxisD,newvecsave[ss])
#        #plotnew[1],=plt.plot(timeaxisD,newvecsave[1])
#        plt.legend([plotnew[0],plotnew[1],plotnew[2],plotnew[3],plotnew[4]], ['new infected','new infected detected','new dead','new recovered','new hospitalized'],fontsize=20)
#        axe0.set_xlabel('time',fontsize=30)
#        plt.savefig("New_VS_time_"+str(sumev[k1]) + '.png')
##        
##        
##        
##        
##        
##        
##        # PLOT OF TOTAL CASES
##        
##        
##        
##        
##        
##        timeaxisD=range(totdays+2)
#        totvecsave=[]
#        for i in range(7,12):
#            yt=0
#            ytd=1
#            if i==7:
#                totvec=[0,totinitinf]
#            else:
#                totvec=[0, 0]
#            #totvec=[0, 0]
#            totvec1=0
#            while yt<len(newchanges[i]):
#                if (yt//dayduration>(ytd-1)):
#                    #j+=1
#                    ytd+=1
#                    #day1=day[j]
#                    totvec.append(newchanges[i][yt])
#        #        else:
#        #            newvec1=newvec1+newchanges[i][yt]
#                #totvec[ytd]=totvec1
#                yt+=1
#            totvecsave.append(totvec)
##        
##        
##        
##        fig1 = plt.figure(figsize=(12.5,7.5))
##        axe1=fig1.gca()
##        for ss in range(5):
##            timeaxisD=range(len(totvecsave[ss]))
##            plotnew1[7+ss],=plt.plot(timeaxisD,totvecsave[ss])
##        #plotnew[1],=plt.plot(timeaxisD,newvecsave[1])
##        plt.legend([plotnew1[7],plotnew1[8],plotnew1[9],plotnew1[10],plotnew1[11]], ['total infected','total infected detected','total dead','total recovered','total hospitalized'],fontsize=20)
##        axe1.set_xlabel('time',fontsize=30)
##        plt.savefig("Total_VS_time_"+str(sumev[k1]) + '.png')
##        
##        
##        print('dead: ',totvecsave[3])
##        print('dead2: ',totvecsave[0])
##        
##        
##        # PLOT OF DAILY CASES
##        
##        
##        
##        
##        
##        timeaxisD=range(totdays+1)
#        dailvecsave=[]
#        for i in range(14,17):
#            yt=0
#            ytd=1
#            if i==14:
#                dailvec=[0,totinitinf]
#            else:
#                dailvec=[0, 0]
#            #dailvec=[0, 0]
#            dailvec1=0
#            while yt<len(newchanges[i]):
#                if (yt//dayduration>(ytd-1)):
#                    #j+=1
#                    ytd+=1
#                    #day1=day[j]
#                    dailvec.append(newchanges[i][yt])
#        #        else:
#        #            newvec1=newvec1+newchanges[i][yt]
#                #totvec[ytd]=totvec1
#                yt+=1
#            dailvecsave.append(dailvec)
#        
#        
#        print(dailvecsave)
#        
#        
#        fig2 = plt.figure(figsize=(12.5,7.5))
#        axe2=fig2.gca()
#        for ss in range(3):
#            timeaxisD=range(len(dailvecsave[ss]))
#            plotnew2[14+ss],=plt.plot(timeaxisD,dailvecsave[ss])
#        #plotnew[1],=plt.plot(timeaxisD,newvecsave[1])
#        plt.legend([plotnew2[14],plotnew2[15],plotnew2[16]], ['daily infected','daily infected detected','daily hospitalized'],fontsize=20)
#        axe2.set_xlabel('time',fontsize=30)
#        plt.savefig("Daily_VS_time_"+str(sumev[k1]) + '.png')
#        #print(dailinfdet)
#        
#        
#        
#        
#        
#        # PLOT OF R0 METHOD 1
#        
#        
#        
#        
#        
#        fig3 = plt.figure(figsize=(12.5,7.5))
#        axe3=fig3.gca()
#        #print(len(newvecsave[0]))
#        #print(len(dailvecsave[0]))
#        rat=[]
#        for q in range(len(dailvecsave[0])-1):
#            if (dailvecsave[0][q]+dailvecsave[2][q])>0:
#                rat.append(newvecsave[0][q+1]/(dailvecsave[0][q]+dailvecsave[2][q]))
#            
#        plotnew3,=plt.plot(range(len(rat)),rat)
#        #plotnew[1],=plt.plot(timeaxisD,newvecsave[1])
#        plt.legend([plotnew3], ['R0'],fontsize=20)
#        axe3.set_xlabel('time',fontsize=30)
#        plt.savefig("R0_VS_time_1_"+str(sumev[k1]) + '.png')
#        #plt.savefig("New_VS_time_"+str(sume) + '.png')
#        
#        #print(dailinf)
#        
#        
#        
#        
#        
#        # PLOT OF R0 METHOD 2
#        
#        
#        
#        
#        GR0=nx.Graph()
#        listaR0=[]
#        for j in range(len(datR0)):
#            listaR0.append(int(float(datR0[j][0])))
#            GR0.add_node(datR0[j][1])
#            GR0.node[datR0[j][1]]['infect']=0
#        
#        #print(GR0.nodes(data=True))
#        #print(listaR0)
#        cc=0
#        infe=[[],[]]
#        for u in range(listaR0[len(listaR0)-1]//dayduration):
#            #print(dayduration*u)
#            #print(listaR0[cc])
#            while (listaR0[cc]<dayduration*u) and (listaR0[cc]>=dayduration*(u-1)):
#                #print(cc)
#                GR0.node[datR0[cc][1]]['infect']+=1
#                cc+=1
#            infe[0].append(u)
#            suma1=0
#            infe1=[v for v in GR0.node if GR0.node[v]['infect']>0]
#            #print(infe1)
#            for rr in infe1:
#                #print(GR0.node[rr]['infect'])
#                suma1=suma1+GR0.node[rr]['infect']
#            if len(infe1)==0:
#                infe[1].append(0)
#            else:
#                infe[1].append(suma1/len(infe1))
#        
#        #print(infe)
#        #print(GR0.nodes(data=True))
#        tim=infe[0]
#        infec=infe[1]
#        
#        
#    
#        
#        
#        
#        fig4 = plt.figure(figsize=(12.5,7.5))
#        axe4=fig4.gca()
#        
#        plotnew4,=plt.plot(tim,infec)
#        #plotnew[1],=plt.plot(timeaxisD,newvecsave[1])
#        plt.legend([plotnew4], ['R0'],fontsize=20)
#        axe4.set_xlabel('time',fontsize=30)
#        plt.savefig("R0_VS_time_2_"+str(sumev[k1]) + '.png')
#        
#        
#        
        #THINGS TO STUDY
        
        #MAX HOSP
        
        compvecMAXHOSP1.append(max(dailvecsave[2]))
        compvecMAXDIE1.append(max(totvecsave[2]))
        compvecMAXINF1.append(max(totvecsave[0]))
        compvecMAXINFDAILY1.append(max(dailvecsave[0]))
        compvecMAXTIME1.append(len(timeaxisD))
    
    compvecMAXHOSP.append(np.mean(compvecMAXHOSP1))
    compvecMAXHOSPvar.append(np.std(compvecMAXHOSP1))
    compvecMAXDIE.append(np.mean(compvecMAXDIE1))
    compvecMAXDIEvar.append(np.std(compvecMAXDIE1))
    compvecMAXINF.append(np.mean(compvecMAXINF1))
    compvecMAXINFvar.append(np.std(compvecMAXINF1))
    compvecMAXINFDAILY.append(np.mean(compvecMAXINFDAILY1))
    compvecMAXINFDAILYvar.append(np.std(compvecMAXINFDAILY1))
    compvecMAXTIME.append(np.mean(compvecMAXTIME1))
    compvecMAXTIMEvar.append(np.std(compvecMAXTIME1))

yaxisVect=[compvecMAXHOSP,compvecMAXDIE,compvecMAXINF,compvecMAXINFDAILY,compvecMAXTIME]
yaxisVectvar=[compvecMAXHOSPvar,compvecMAXDIEvar,compvecMAXINFvar,compvecMAXINFDAILYvar,compvecMAXTIMEvar]

#print(yaxisVectvar)

names=['Maximum Hospitalized','Total dead','Total infected','Maximum daily infected','Time for pandemia to finish']
namesfig=['Maximum_Hospitalized','Total_dead','Total_infected','Maximum_daily_infected','Time_for_pandemia_to_finish']


plotinteresv=[]
for ss1 in range(len(Vecs)):
    if len(Vecs[ss1])>1:
        plotinteresv.append(ss1)


#Vecs=[Nv,nov,n2v,n3v,n4v,n5v,pininfv,daydurationv,prewirev,pv,tsocdistv,tsocdistliftv,psocdistv,pov,posocdistv,R01v,pdv,quarantinev,pkv,pmobility1v,pmobilitysocdistv,pmobilityafterv,mobfreqv,tautempfreev,tautempplacev,tautemphospv,tautempfreequav,tautempplacequav,tautemphospquav,tautempfreeafterquav,tautempplaceafterquav,tautemphospafterquav,tausick1v,sigmasickv,taurecovered1v,sigmarecoveredv,tauinfected1v,sigmainfectedv,tauhosp1v,sigmahospv,taurec1v,sigmarecv,mobilityv]    


itemv=[]
namev=[]
for u1 in range(len(plotinteresv)):
    plotinteres=plotinteresv[u1]
    if plotinteres==0:
        item='N'
        name='N'
    if plotinteres==1:
        item='number of places'
        name='number_of_places'
    if plotinteres==2:
        item='percentage of 2-families'
        name='percentage_of_2_families'
    if plotinteres==3:
        item='percentage of 3-families'
        name='percentage_of_3_families'
    if plotinteres==4:
        item='percentage of 4-families'
        name='percentage_of_4_families'
    if plotinteres==5:
        item='percentage of 5-families'
        name='percentage_of_5_families'
    if plotinteres==6:
        item='percentage of initially infected'
        name='percentage_of_initially_infected'
    if plotinteres==7:
        item='dayduration'
        name='dayduration'
    if plotinteres==8:
        item='p of rewiring in small-world'
        name='p_of_rewiring_in_small_world'
    if plotinteres==9:
        item='p of accepted personal links'
        name='p_of_accepted_personal_links'
    if plotinteres==10:
        item='time of social distancing'
        name='time_of_social_distancing'
    if plotinteres==11:
        item='time of social distancing lifted'
        name='time_of_social_distancing_lifted'
    if plotinteres==12:
        item='percentage of personal links dropped'
        name='percentage_of_personal_links_dropped'
    if plotinteres==13:
        item='percentage of place links accepted'
        name='percentage_of_place_links_accepted'
    if plotinteres==14:
        item='percentage of place links dropped'
        name='percentage_of_place_links_dropped'
    if plotinteres==15:
        item='R01'
        name='R01'
    if plotinteres==16:
        item='probability of detection'
        name='probability_of_detection'
    if plotinteres==17:
        item='quarantine'
        name='quarantine'
    if plotinteres==18:
        item='p of nearest neighbors in small-world'
        name='nearest_neighbors_small_world'
    if plotinteres==19:
        item='prob. to rewire'
        name='prob_to_rewire'
    if plotinteres==20:
        item='prob. to rewire in social distancing'
        name='prob_to_rewire_social_dist'
    if plotinteres==21:
        item='prob. to rewire after social distancing'
        name='prob_to_rewire_after_social_dist'
    if plotinteres==22:
        item='frequency of rewiring'
        name='frequency_of_rewiring'
    if plotinteres==23:
        item='average free temperature'
        name='av_free_temp'
    if plotinteres==24:
        item='average temperature of places'
        name='av_place_temp'
    if plotinteres==25:
        item='average hospital temperature'
        name='av_hosp_temp'
    if plotinteres==26:
        item='average free temperature in social distancing'
        name='av_free_temp_soc_dist'
    if plotinteres==27:
        item='average place temperature in social distancing'
        name='av_place_temp_soc_dist'
    if plotinteres==28:
        item='average hospital temperature in social distancing'
        name='av_hosp_temp_soc_dist'
    if plotinteres==29:
        item='average free temperature after social distancing'
        name='av_free_temp_after_soc_dist'
    if plotinteres==30:
        item='average place temperature after social distancing'
        name='av_place_temp_after_soc_dist'
    if plotinteres==31:
        item='average hospital temperature after social distancing'
        name='av_hosp_temp_after_soc_dist'
    if plotinteres==32:
        item='average time to die in hospital'
        name='av_time_death'
    if plotinteres==33:
        item='variance time to die in hospital'
        name='var_time_death'
    if plotinteres==34:
        item='average time for recovered to susceptible'
        name='av_time_rec_to_susc'
    if plotinteres==35:
        item='variance time for recovered to susceptible'
        name='var_time_rec_to_susc'
    if plotinteres==36:
        item='average time for infected to recover'
        name='av_time_inf_to_rec'
    if plotinteres==37:
        item='variance time for infected to recover'
        name='var_time_inf_to_rec'
    if plotinteres==38:
        item='average time for hospitalized to recover'
        name='av_time_hosp_to_rec'
    if plotinteres==39:
        item='variance time for hospitalized to recover'
        name='var_time_hosp_to_rec'
    if plotinteres==40:
        item='average time for infected to hospitalized'
        name='av_time_inf_to_hosp'
    if plotinteres==41:
        item='variance time for infected to hospitalized'
        name='var_time_inf_to_hosp'
    if plotinteres==42:
        item='mobility'
        name='mobility'

        
    itemv.append(item)
    namev.append(name)









if len(plotinteresv)>0:
    for u in range(len(yaxisVect)):
        figxaxis=Vecs[plotinteresv[0]]
        figyaxis=yaxisVect[u]
        figerr=yaxisVectvar[u] 
    
        figv = plt.figure(figsize=(12.5,7.5))
        axev=figv.gca()
        
        itemfig=namesfig[u]+'_VS_' + namev[0]
        plt.title(names[u]+' VS ' + itemv[0],size=30)
        axev.set_ylabel(names[u],fontsize=30)
    
        axev.set_xlabel(itemv[0],fontsize=30)
        plotp=plt.plot(figxaxis,figyaxis)
        axev.errorbar(figxaxis,figyaxis,yerr=np.multiply(figerr,1))
        plt.rcParams.update({'font.size': 22})
        plt.savefig(outputfile+itemfig+"_"+str(sum(sumev)) + '.png')
        plt.show()
    


out_string_Par=""
out_string_Par+="N"+","+str(Nv)
out_string_Par+="\n"
out_string_Par+="No"+","+str(nov)
out_string_Par+="\n"
out_string_Par+="n2"+","+str(n2v)
out_string_Par+="\n"
out_string_Par+="n3"+","+str(n3v)
out_string_Par+="\n"
out_string_Par+="n4"+","+str(n4v)
out_string_Par+="\n"
out_string_Par+="n5"+","+str(n5v)
out_string_Par+="\n"
out_string_Par+="pininf"+","+str(pininfv)
out_string_Par+="\n"
out_string_Par+="dayduration"+","+str(daydurationv)
out_string_Par+="\n"
out_string_Par+="prewire"+","+str(prewirev)
out_string_Par+="\n"
out_string_Par+="p"+","+str(pv)
out_string_Par+="\n"
out_string_Par+="tsocdist"+","+str(tsocdistv)
out_string_Par+="\n"
out_string_Par+="tsocdistlift"+","+str(tsocdistliftv)
out_string_Par+="\n"
out_string_Par+="psocdist"+","+str(psocdistv)
out_string_Par+="\n"
out_string_Par+="po"+","+str(pov)
out_string_Par+="\n"
out_string_Par+="posocdist"+","+str(posocdistv)
out_string_Par+="\n"
out_string_Par+="R01"+","+str(R01v)
out_string_Par+="\n"
out_string_Par+="pd"+","+str(pdv)
out_string_Par+="\n"
out_string_Par+="quarantine"+","+str(quarantinev)
out_string_Par+="\n"
out_string_Par+="pk"+","+str(pkv)
out_string_Par+="\n"
out_string_Par+="pmobility1"+","+str(pmobility1v)
out_string_Par+="\n"
out_string_Par+="pmobilitysocdist"+","+str(pmobilitysocdistv)
out_string_Par+="\n"
out_string_Par+="pmobilityafter"+","+str(pmobilityafterv)
out_string_Par+="\n"
out_string_Par+="mobility"+","+str(mobilityv)
out_string_Par+="\n"
out_string_Par+="mobfreq"+","+str(mobfreqv)
out_string_Par+="\n"
out_string_Par+="tautempfree"+","+str(tautempfreev)
out_string_Par+="\n"
out_string_Par+="tautempplace"+","+str(tautempplacev)
out_string_Par+="\n"
out_string_Par+="tautemphosp"+","+str(tautemphospv)
out_string_Par+="\n"
out_string_Par+="tautempfreequa"+","+str(tautempfreequav)
out_string_Par+="\n"
out_string_Par+="tautempplacequa"+","+str(tautempplacequav)
out_string_Par+="\n"
out_string_Par+="tautemphospqua"+","+str(tautemphospquav)
out_string_Par+="\n"
out_string_Par+="tautempfreeafterqua"+","+str(tautempfreeafterquav)
out_string_Par+="\n"
out_string_Par+="tautempplaceafterqua"+","+str(tautempplaceafterquav)
out_string_Par+="\n"
out_string_Par+="tautemphospafterqua"+","+str(tautemphospafterquav)
out_string_Par+="\n"
out_string_Par+="tausick1"+","+str(tausick1v)
out_string_Par+="\n"
out_string_Par+="sigmasick"+","+str(sigmasickv)
out_string_Par+="\n"
out_string_Par+="taurecovered1"+","+str(taurecovered1v)
out_string_Par+="\n"
out_string_Par+="sigmarecovered"+","+str(sigmarecoveredv)
out_string_Par+="\n"
out_string_Par+="tauinfected1"+","+str(tauinfected1v)
out_string_Par+="\n"
out_string_Par+="sigmainfected"+","+str(sigmainfectedv)
out_string_Par+="\n"
out_string_Par+="taurec1"+","+str(taurec1v)
out_string_Par+="\n"
out_string_Par+="sigmarec"+","+str(sigmarecv)
out_string_Par+="\n"
out_string_Par+="tauhosp1"+","+str(tauhosp1v)
out_string_Par+="\n"
out_string_Par+="sigmahosp"+","+str(sigmahospv)
out_string_Par+="\n"


#outfilename=r"/Users/Chris/Documents/Epidemiology_project/New_data"

out_file_Par=open("Par_sume_"+str(sum(sumev))+".csv","w+")

out_file_Par.write(out_string_Par)
out_file_Par.close()

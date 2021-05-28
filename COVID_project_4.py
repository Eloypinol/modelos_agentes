# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:58:02 2020

@author: Chris
"""
from time import time as tmm

outfilename=r"results"

tiempo_inicial = tmm() 

import argparse

parser = argparse.ArgumentParser(description='Epidemiology project')


parser.add_argument('-arg1','--N',type=int,help='typewanted')
parser.add_argument('-arg2','--numberofplaces',type=int,help='meas')
parser.add_argument('-arg3','--n2',type=float,help='realiz')
parser.add_argument('-arg4','--n3',type=float,help='training')
parser.add_argument('-arg5','--n4',type=float,help='test')
parser.add_argument('-arg6','--n5',type=float,help='n_washing')
parser.add_argument('-arg7','--pininf', type=float,help='Nv')
parser.add_argument('-arg8','--dayduration', type=int,help='ncv')
parser.add_argument('-arg9','--prewire', type=float,help='typewanted')
parser.add_argument('-arg10','--p', type=float,help='typewanted')
parser.add_argument('-arg11','--tsocdist', type=int,help='typewanted')
parser.add_argument('-arg12','--tsocdistlift', type=int,help='typewanted')
parser.add_argument('-arg13','--psocdist', type=float,help='typewanted')
parser.add_argument('-arg14','--po', type=float,help='typewanted')
parser.add_argument('-arg15','--posocdist', type=float,help='typewanted')
parser.add_argument('-arg16','--R01', type=int,help='typewanted')
parser.add_argument('-arg17','--pd', type=float,help='typewanted')
parser.add_argument('-arg18','--quarantine', type=int,help='typewanted')
parser.add_argument('-arg19','--pk', type=float,help='typewanted')
parser.add_argument('-arg20','--pmobility1',  type=float,help='typewanted')
parser.add_argument('-arg21','--pmobilitysocdist',  type=float,help='typewanted')
parser.add_argument('-arg22','--pmobilityafter',  type=float,help='typewanted')
parser.add_argument('-arg23','--mobility',  type=int,help='typewanted')
parser.add_argument('-arg24','--mobfreq', type=int,help='typewanted')
parser.add_argument('-arg25','--tautempfree',  type=float,help='typewanted')
parser.add_argument('-arg26','--tautempplace', type=float,help='typewanted')
parser.add_argument('-arg27','--tautemphosp', type=float,help='typewanted')
parser.add_argument('-arg28','--tautempfreequa', type=float,help='typewanted')
parser.add_argument('-arg29','--tautempplacequa', type=float,help='typewanted')
parser.add_argument('-arg30','--tautemphospqua',  type=float,help='typewanted')
parser.add_argument('-arg31','--tautempfreeafterqua', type=float,help='typewanted')
parser.add_argument('-arg32','--tautempplaceafterqua', type=float,help='typewanted')
parser.add_argument('-arg33','--tautemphospafterqua',  type=float,help='typewanted')
parser.add_argument('-arg34','--tausick1', type=float,help='typewanted')
parser.add_argument('-arg35','--sigmasick', type=float,help='typewanted')
parser.add_argument('-arg36','--taurecovered1', type=float,help='typewanted')
parser.add_argument('-arg37','--sigmarecovered', type=float,help='typewanted')
parser.add_argument('-arg38','--tauinfected1', type=float,help='typewanted')
parser.add_argument('-arg39','--sigmainfected', type=float,help='typewanted')
parser.add_argument('-arg40','--taurec1', type=float,help='typewanted')
parser.add_argument('-arg41','--sigmarec', type=float,help='typewanted')
parser.add_argument('-arg42','--tauhosp1', type=float,help='typewanted')
parser.add_argument('-arg43','--sigmahosp', type=float,help='typewanted')
parser.add_argument('-arg44','--resuming',type=int,help='typewanted')
parser.add_argument('-arg45','--accur',type=int,help='typewanted')
parser.add_argument('-arg46','--runn',type=int,help='typewanted')


args=parser.parse_args()

#python COVID_project_4.py --N 500 --numberofplaces 10 --n2 0.25 --n3 0.25 --n4 0.25 --n5 0.015 --pininf 0.01 --dayduration 250 --prewire 0.5 --p 0.06 --tsocdist 6 --tsocdistlift 45 --psocdist 0.85 --po 0.06 --posocdist 0.85 --R01 40 --pd 0.1 --quarantine 0 --pk 0.2 --pmobility1 1E-3 --pmobilitysocdist 1E-4 --pmobilityafter 5E-4 --mobility 0 --mobfreq 1 --tautempfree 1E-3 --tautempplace 1E-3 --tautemphosp 1E-3 --tautempfreequa 1E-11 --tautempplacequa 1E-10 --tautemphospqua 0.01 --tautempfreeafterqua 1E-7 --tautempplaceafterqua 1E-6 --tautemphospafterqua 1E-5 --tausick1 45 --sigmasick 9 --taurecovered1 110 --sigmarecovered 13 --tauinfected1 14 --sigmainfected 2 --taurec1 17 --sigmarec 6 --tauhosp1 36 --sigmahosp 24 --resuming 0 --accur 5 --runn 0


import numpy as np

import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import networkx as nx

import math

#import itertools

import time

#import scipy.special as sps

#from scipy.stats import gamma

#from scipy.stats import poisson

from math import log10, floor

from func_4 import *

import csv

#import pandas as pd

#with open("Par_Tot_pop_"+str(N)+"_p_"+str(p)+"_po_"+str(po)+"_tausick_"+str(tausick1)+"_taurec_"+str(taurecovered1)+"_tauinf_"+str(tauinfected1)+"_beta_"+str(beta)+"_pInitial_"+str(pininf)+"_pd_"+str(pd)+"_tsocdist_"+str(tsocdist)+"_dayduration_"+str(dayduration)+"_psocdist_"+str(psocdist)+"_posocdist_"+str(posocdist)+".csv","w") as out_file_Par:
#out_file_Par=open("Par_Tot_pop_"+str(N)+"_p_"+str(p)+"_po_"+str(po)+"_tausick_"+str(tausick1)+"_taurec_"+str(taurecovered1)+"_tauinf_"+str(tauinfected1)+"_beta_"+str(beta)+"_pInitial_"+str(pininf)+"_pd_"+str(pd)+"_tsocdist_"+str(tsocdist)+"_dayduration_"+str(dayduration)+"_psocdist_"+str(psocdist)+"_posocdist_"+str(posocdist)+".csv","w")
#parameters

cluster=0

round_to_n = lambda x, n: round(x, -int(floor(log10(x))) + (n - 1))


if cluster==0:

    resuming=0
    
    N=500 #total population

    numberofplaces=10 #total number of places (hospital,transport,work)
    print('Total population: ',N-numberofplaces)
    print('Number of places: ',numberofplaces)
    n2=0.25 #percentage of 2 person families
    n3=0.25 #percentage of 3 person families
    n4=0.25 #percentage of 4 person families
    n5=0.015 #percentage of 5 person families
    #rest is single person groups
    
    pininf=0.01 #percentage of initially infected people
    print('Number of initially infected people: ',math.ceil(pininf*N))
    dayduration=250#math.ceil(0.1*N) #how many steps in a day
    prewire=0.5 #probbility to rewire an edge for the small-world graph of the connections between people of different groups
    print('Probability to rewire in small-world: ',prewire)
    p=0.06# probability to accept an edge of the small world-graph  #probabilidad conexiones entre personas
    print('Small-world parameter: ',p)
    tsocdist=6*dayduration#5000*dayduration #time at which social distancing measures are implemented
    print('Day social distancing will be imposed: ',tsocdist/dayduration)
    tsocdistlift=45*dayduration
    print('Day social distancing will be lifted: ',tsocdistlift/dayduration)
    psocdist=0.85 #percentage of social connections removed
    print('Percentage of social connections removed: ',psocdist)
    po=0.06# probability of a connection between a citizen and a place 
    print('Probability of a connection between a citizen and a place: ',po)
    posocdist=0.85 # probability to remove these connections between people and places after quarantine measures
    print('Probability to remove these connections between people and places after quarantine measures: ',posocdist)
    #ph=0.005 #probability of getting hospitalized at each time step after being infected
    #pr=TIME FROM ADMISSION TO DISCHARGE 0.05 #probability to recover at each time step
    R01=40 #number of people a person can infect

    pd=0.1 #probability of an infected person to be detected
    print('Probability of an infected person to be detected: ',pd)
    quarantine=0 #whether we start with a quarantine setup or not (0 is deactivated, 1 is activated)
    
    #beta=10**-6 #beta of the ising model
    
    
    N2=math.ceil(N*n2)//2;
    print('Number of houses with 2-persons: ',N2)
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
    pk=0.2
    knearestneighbors=math.ceil(pk*NTotH)
    print('knearestneighbors in small-world: ',knearestneighbors)
    pmobility1=0.001
    print('Probability to switch neighbor if mobility is allowed: ',pmobility1)
    pmobilitysocdist=0.0001
    print('Probability to switch neighbor if mobility is allowed under social distancing measures: ',pmobilitysocdist)
    pmobilityafter=0.0005
    print('Probability to switch neighbor if mobility is allowed after social distancing measures: ',pmobilityafter)


    mobility=0
    print('Is mobility allowed? ',mobility)
    
    mobfreq=math.ceil(dayduration/250)
    print('Frequency of switching neighbors if mobility is allowed: ',mobfreq)
    
    tautempfree=round_to_n(1*10**-4,1)
    print('Average contagious temperature among people: ',tautempfree)
    
    tautempplace=round_to_n(1*10**-3,1)
    print('Average contagious temperature in places: ',tautempplace)
    
    tautemphosp=round_to_n(1*10**-3,1)
    print('Average contagious temperature in hospital: ',tautemphosp)
    
    tautempfreequa=round_to_n(1*10**-11,1)
    print('Average contagious temperature among people in quarantine conditions: ',tautempfreequa)

    tautempplacequa=round_to_n(1*10**-10,1)
    print('Average contagious temperature in places in quarantine conditions: ',tautempplacequa)
    
    tautemphospqua=round_to_n(1*10**-2,1)
    print('Average contagious temperature in hospital in quarantine conditions: ',tautemphospqua)
    
    tautempfreeafterqua=round_to_n(1*10**-7,1)
    print('Average contagious temperature among people after quarantine conditions: ',tautempfreeafterqua)

    tautempplaceafterqua=round_to_n(1*10**-6,1)
    print('Average contagious temperature in places after quarantine conditions: ',tautempplaceafterqua)

    tautemphospafterqua=round_to_n(1*10**-5,1)
    print('Average contagious temperature in hospital after quarantine conditions: ',tautemphospafterqua)
    
    
    
    tausick1=round_to_n(1.25*36*dayduration,1)#*dayduration #average time a person stays sick in the hospital (normal dist)
    print('Average time a person stays sick in the hospital before dying: ',tausick1)
    sigmasick=(0.015*573*dayduration)**2 #standard deviation of sick time
    print('Variance of time a person stays sick in the hospital before dying: ',sigmasick)
    alphasick=tausick1**2/sigmasick
    betasick=sigmasick/tausick1
    
    
    taurecovered1=round_to_n(0.3*365*dayduration,1)#*dayduration #average time it takes for a person to recover (normal dist)
    print('Average time it takes for a recovered person to become susceptible: ',taurecovered1)
    sigmarecovered=(0.035*365*dayduration)**2 #standard deviation of recovery time
    print('Variance of time it takes for a recovered person to become susceptible: ',sigmarecovered)
    alpharecovered=taurecovered1**2/sigmarecovered
    betarecovered=sigmarecovered/taurecovered1
    
    
    tauinfected1=round_to_n(2*7*dayduration,1)#*dayduration #average time a person stays infected but without symptoms (normal dist) (just carrying the virus)
    print('Average time a person stays infected before recovering: ',tauinfected1)
    sigmainfected=(0.1*17*dayduration)**2 #standard deviation of time being infected
    print('Variance of time a person stays infected before recovering: ',sigmainfected)
    alphainfected=tauinfected1**2/sigmainfected
    betainfected=sigmainfected/tauinfected1
    
    
    taurec1=round_to_n(0.75*23.5*dayduration,1)#*dayduration #average time it takes for a person to recover (normal dist)
    print('Average time for a person to recover from hospital: ',taurec1)
    sigmarec=(0.1*62*dayduration)**2 #standard deviation of recovery time
    print('Variance of time for a person to recover from hospital: ',sigmarec)
    alpharec=taurec1**2/sigmarec
    betarec=sigmarec/taurec1
    
    
    tauhosp1=0.75*48.5*dayduration#*dayduration #average time it takes for a person to recover (normal dist)
    print('Average time it takes for infected person to go to hospital: ',tauhosp1)
    sigmahosp=(2*12*dayduration)**2 #standard deviation of recovery time
    print('Variance of time it takes for infected person to go to hospital: ',sigmahosp)
    alphahosp=tauhosp1**2/sigmahosp
    betahosp=sigmahosp/tauhosp1
    
    accur=5
    
    runn=0

#    outfilename=r"/Users/Chris/Documents/Epidemiology_project/New_data"

else:

    resuming=args.resuming
    N=args.N
    
    numberofplaces=args.numberofplaces #total number of places (hospital,transport,work)
    n2=args.n2 #percentage of 2 person families
    n3=args.n3 #percentage of 3 person families
    n4=args.n4 #percentage of 4 person families
    n5=args.n5 #percentage of 5 person families
    #rest is single person groups
    
    pininf=args.pininf #percentage of initially infected people
    dayduration=args.dayduration#math.ceil(0.1*N) #how many steps in a day
    prewire=args.prewire #probbility to rewire an edge for the small-world graph of the connections between people of different groups
    p=args.p# probability to accept an edge of the small world-graph 
    tsocdist=args.tsocdist*dayduration#5000*dayduration #time at which social distancing measures are implemented
    tsocdistlift=args.tsocdistlift*dayduration
    psocdist=args.psocdist #percentage of social connections removed
    po=args.po# probability of a connection between a citizen and a place 
    posocdist=args.posocdist # probability to remove these connections between people and places after quarantine measures
    R01=args.R01 #number of people a person can infect

    pd=args.pd #probability of an infected person to be detected
    quarantine=args.quarantine #whether we start with a quarantine setup or not (0 is deactivated, 1 is activated)

    N2=math.ceil(N*n2)//2;
    N3=(math.ceil(N*n3))//3;
    N4=(math.ceil(N*n4))//4;
    N5=math.ceil(N*n5)//5;
    No=numberofplaces;
    N1=N-5*N5-2*N2-3*N3-4*N4-No;
        
    NTotH=N1+N2+N3+N4+N5 #total number of houses

    pk=args.pk
    knearestneighbors=math.ceil(pk*NTotH)
    pmobility1=args.pmobility1
    pmobilitysocdist=args.pmobilitysocdist
    pmobilityafter=args.pmobilityafter

    mobility=args.mobility
    
    mobfreq=args.mobfreq
    
    tautempfree=args.tautempfree

    tautempplace=args.tautempplace
 
    tautemphosp=args.tautemphosp

    tautempfreequa=args.tautempfreequa
    
    tautempplacequa=args.tautempplacequa
    
    tautemphospqua=args.tautemphospqua
    
    tautempfreeafterqua=args.tautempfreeafterqua

    tautempplaceafterqua=args.tautempplaceafterqua
    
    tautemphospafterqua=args.tautemphospafterqua
    
    
    tausick1=round_to_n(args.tausick1*dayduration,1)#*dayduration #average time a person stays sick in the hospital (normal dist)
    sigmasick=(args.sigmasick*dayduration)**2 #standard deviation of sick time
    alphasick=tausick1**2/sigmasick
    betasick=sigmasick/tausick1
    
    
    taurecovered1=round_to_n(args.taurecovered1*dayduration,1)#*dayduration #average time it takes for a person to recover (normal dist)
    sigmarecovered=(args.sigmarecovered*dayduration)**2 #standard deviation of recovery time
    alpharecovered=taurecovered1**2/sigmarecovered
    betarecovered=sigmarecovered/taurecovered1
    
    
    tauinfected1=round_to_n(args.tauinfected1*dayduration,1)#*dayduration #average time a person stays infected but without symptoms (normal dist) (just carrying the virus)
    sigmainfected=(args.sigmainfected*dayduration)**2 #standard deviation of time being infected
    alphainfected=tauinfected1**2/sigmainfected
    betainfected=sigmainfected/tauinfected1
    
    
    taurec1=round_to_n(args.taurec1*dayduration,1)#*dayduration #average time it takes for a person to recover (normal dist)
    sigmarec=(args.sigmarec*dayduration)**2 #standard deviation of recovery time
    alpharec=taurec1**2/sigmarec
    betarec=sigmarec/taurec1
    
    
    tauhosp1=round_to_n(args.tauhosp1*dayduration,1)#*dayduration #average time it takes for a person to recover (normal dist)
    sigmahosp=(args.sigmahosp*dayduration)**2 #standard deviation of recovery time
    alphahosp=tauhosp1**2/sigmahosp
    betahosp=sigmahosp/tauhosp1

    runn=args.runn
    
    accur=args.accur
    
#    outfilename=r"/home/christos/Desktop/Epidemiology_project/New_data"
#    outfilename=r"New_data"
    
sumvec=[N,numberofplaces,n2,n3,n4,n5,pininf,dayduration,prewire,p,tsocdist,tsocdistlift,psocdist,po,posocdist,R01,pd,quarantine,pk,pmobility1,pmobilitysocdist,pmobilityafter,mobfreq,tautempfree,tautempplace,tautemphosp,tautempfreequa,tautempplacequa,tautemphospqua,tautempfreeafterqua,tautempplaceafterqua,tautemphospafterqua,alphasick,betasick,alpharecovered,betarecovered,alphainfected,betainfected,alphahosp,betahosp,alpharec,betarec]    
    
sume=0
for i in range(len(sumvec)):
    sume+=(i+1)*sumvec[i]



tautemp=[tautempfree,tautempplace,tautemphosp,tautempfreequa,tautempplacequa,tautemphospqua,tautempfreeafterqua,tautempplaceafterqua,tautemphospafterqua]
sigmatemp=[]
alphatemp=[]
betatemp=[]
for u in range(len(tautemp)):
    sigmatemp.append(tautemp[u]**2)
    alphatemp.append(tautemp[u]**2/sigmatemp[u])
    betatemp.append(sigmatemp[u]/tautemp[u])


betafree=random.gammavariate(alphatemp[0],betatemp[0])

#print('betafree',betafree)



out_string_Par=""
out_string_Par+="N"+","+str(N)
out_string_Par+="\n"
out_string_Par+="No"+","+str(numberofplaces)
out_string_Par+="\n"
out_string_Par+="n2"+","+str(n2)
out_string_Par+="\n"
out_string_Par+="n3"+","+str(n3)
out_string_Par+="\n"
out_string_Par+="n4"+","+str(n4)
out_string_Par+="\n"
out_string_Par+="n5"+","+str(n5)
out_string_Par+="\n"
out_string_Par+="pininf"+","+str(pininf)
out_string_Par+="\n"
out_string_Par+="dayduration"+","+str(dayduration)
out_string_Par+="\n"
out_string_Par+="prewire"+","+str(prewire)
out_string_Par+="\n"
out_string_Par+="p"+","+str(p)
out_string_Par+="\n"
out_string_Par+="tsocdist"+","+str(tsocdist/dayduration)
out_string_Par+="\n"
out_string_Par+="tsocdistlift"+","+str(tsocdistlift/dayduration)
out_string_Par+="\n"
out_string_Par+="psocdist"+","+str(psocdist)
out_string_Par+="\n"
out_string_Par+="po"+","+str(po)
out_string_Par+="\n"
out_string_Par+="posocdist"+","+str(posocdist)
out_string_Par+="\n"
out_string_Par+="R01"+","+str(R01)
out_string_Par+="\n"
out_string_Par+="pd"+","+str(pd)
out_string_Par+="\n"
out_string_Par+="quarantine"+","+str(quarantine)
out_string_Par+="\n"
out_string_Par+="pk"+","+str(pk)
out_string_Par+="\n"
out_string_Par+="pmobility1"+","+str(pmobility1)
out_string_Par+="\n"
out_string_Par+="pmobilitysocdist"+","+str(pmobilitysocdist)
out_string_Par+="\n"
out_string_Par+="pmobilityafter"+","+str(pmobilityafter)
out_string_Par+="\n"
out_string_Par+="mobility"+","+str(mobility)
out_string_Par+="\n"
out_string_Par+="mobfreq"+","+str(mobfreq)
out_string_Par+="\n"
out_string_Par+="tautempfree"+","+str(tautempfree)
out_string_Par+="\n"
out_string_Par+="tautempplace"+","+str(tautempplace)
out_string_Par+="\n"
out_string_Par+="tautemphosp"+","+str(tautemphosp)
out_string_Par+="\n"
out_string_Par+="tautempfreequa"+","+str(tautempfreequa)
out_string_Par+="\n"
out_string_Par+="tautempplacequa"+","+str(tautempplacequa)
out_string_Par+="\n"
out_string_Par+="tautemphospqua"+","+str(tautemphospqua)
out_string_Par+="\n"
out_string_Par+="tautempfreeafterqua"+","+str(tautempfreeafterqua)
out_string_Par+="\n"
out_string_Par+="tautempplaceafterqua"+","+str(tautempplaceafterqua)
out_string_Par+="\n"
out_string_Par+="tautemphospafterqua"+","+str(tautemphospafterqua)
out_string_Par+="\n"
out_string_Par+="tausick1"+","+str(tausick1/dayduration)
out_string_Par+="\n"
out_string_Par+="sigmasick"+","+str(np.sqrt(sigmasick)/dayduration)
out_string_Par+="\n"
out_string_Par+="taurecovered1"+","+str(taurecovered1/dayduration)
out_string_Par+="\n"
out_string_Par+="sigmarecovered"+","+str(np.sqrt(sigmarecovered)/dayduration)
out_string_Par+="\n"
out_string_Par+="tauinfected1"+","+str(tauinfected1/dayduration)
out_string_Par+="\n"
out_string_Par+="sigmainfected"+","+str(np.sqrt(sigmainfected)/dayduration)
out_string_Par+="\n"
out_string_Par+="taurec1"+","+str(taurec1/dayduration)
out_string_Par+="\n"
out_string_Par+="sigmarec"+","+str(np.sqrt(sigmarec)/dayduration)
out_string_Par+="\n"
out_string_Par+="tauhosp1"+","+str(tauhosp1/dayduration)
out_string_Par+="\n"
out_string_Par+="sigmahosp"+","+str(np.sqrt(sigmahosp)/dayduration)
out_string_Par+="\n"
out_string_Par+="accur"+","+str(accur)
out_string_Par+="\n"

#end of parameters


GTot,Gromerged,popmerged,Go,Gpop,Gr1,Gr2,Gr3,Gr4,Gr5,nlistmerged,GSW=Constructing(N,n2,n3,n4,n5,No,p,po,knearestneighbors,prewire)


GTot1=GTot.copy() #created this copy because wanted to plot two graphs, one with the groups/families and one with the infected







GTot1,out_string_Sim=initializing(GTot,GTot1,Go,Gpop,pininf,N,popmerged,alphatemp,betatemp,R01,alphasick,betasick,alpharecovered,betarecovered,alphainfected,betainfected,alphahosp,betahosp,alpharec,betarec,resuming,cluster)

if resuming==0:
    out_file_Sim=open(outfilename+"/Sim_sume_"+str(sume)+"_tsocdist_"+str(tsocdist)+"_tsocdistlift_"+str(tsocdistlift)+"_mobility_"+str(mobility)+"_resuming_"+str(resuming)+"_runn_"+str(runn)+".csv","w+")
    out_file_Sim.write(out_string_Sim)
    out_file_R0=open(outfilename+"/R0_sume_"+str(sume)+"_tsocdist_"+str(tsocdist)+"_tsocdistlift_"+str(tsocdistlift)+"_mobility_"+str(mobility)+"_resuming_"+str(resuming)+"_runn_"+str(runn)+".csv","w+")
    #out_file_Sim.write("")
    #out_file_R0.close()
else:
    out_file_Sim=open(outfilename+"/Sim_sume_"+str(sume)+"_tsocdist_"+str(tsocdist)+"_tsocdistlift_"+str(tsocdistlift)+"_mobility_"+str(mobility)+"_resuming_"+str(resuming)+"_runn_"+str(runn)+".csv","w+")
    out_file_R0=open(outfilename+"/R0_sume_"+str(sume)+"_tsocdist_"+str(tsocdist)+"_tsocdistlift_"+str(tsocdistlift)+"_mobility_"+str(mobility)+"_resuming_"+str(resuming)+"_runn_"+str(runn)+".csv","w+")

out_file_R0.close()
out_file_Sim.close()

GTot,GTot1=init_plot(GTot1,GTot,R01,alphatemp,betatemp,resuming,cluster)


#seperating population list from total
Gpopu=nx.Graph()
for i in GTot1.nodes():
    if i.find('F')==0:
        Gpopu.add_node(i)

if cluster==0:
    #choosing layot for our network
    pos=nx.spring_layout(GTot)
    
    #plotting networks
    if resuming==0:
        fig1 = plt.figure(figsize=(20,12))
        plt.title('Network structure of population',size=30)
        nx.draw_networkx_nodes(GTot,pos,node_size=[importance for importance in nx.get_node_attributes(GTot,'importance').values()],node_color=[color for color in nx.get_node_attributes(GTot,'color').values()],node_shape='o')
        nx.draw_networkx_labels(GTot,pos)
        nx.draw_networkx_edges(GTot,pos,alpha=0.1)
        
        fig2 = plt.figure(figsize=(10,6))
        plt.title('Initial state of network of infected etc',size=30)
        nx.draw_networkx_nodes(GTot1,pos,node_size=[importance for importance in nx.get_node_attributes(GTot1,'importance').values()],node_color=[color for color in nx.get_node_attributes(GTot1,'color').values()],node_shape='o')
        nx.draw_networkx_edges(GTot1,pos,alpha=0.5,width=[size for size in nx.get_edge_attributes(GTot1,'size').values()],edge_color=[color for color in nx.get_edge_attributes(GTot1,'color').values()])


#time.sleep(5)

alph=1
numdead=0
check=0
numinf=1
numrec=0
numhosmax=0
numhos=0
numcont=1
t=1
tothosp=[]
totrec=[]
totinf=[]
totdead=[]
totaldead=[]
totalrec=[]
totcont=[]
newdead=0
newcont=1
totrec1=0

newinf=0
newdead=0
newrec=0
newrecdet=0
newhosp=0
newinfdet=0

newinfected=[0]
newinfecteddetected=[0]
newdead1=[0]
newhospitalized=[0]
newrecovered=[0]
newrecovereddetected=[0]

totinfected=[0]
totinfecteddetected=[0]
totdead=[0]
totrec=[0]
tothosp=[0]
totrecovereddetected=[0]

dailyinf=[0]
dailyinfdet=[0]
dailyrecdet=[0]
dailyrec=[0]
dailyhosp=[0]
update=0

if resuming==0:
    out_file_Par=open(outfilename+"/Par_sume_"+str(sume)+"_tsocdist_"+str(tsocdist)+"_tsocdistlift_"+str(tsocdistlift)+"_mobility_"+str(mobility)+"_resuming_"+str(resuming)+"_runn_"+str(runn)+".csv","w+")

    out_file_Par.write(out_string_Par)
    out_file_Par.close()

tausick2=nx.get_node_attributes(GTot1,'tausick')
taurecovered2=nx.get_node_attributes(GTot1,'taurecovered')
tauinfected2=nx.get_node_attributes(GTot1,'tauinfected')
gender2=nx.get_node_attributes(GTot1,'gender')
age2=nx.get_node_attributes(GTot1,'age')
state2=nx.get_node_attributes(GTot1,'state')
spin2=nx.get_node_attributes(GTot1,'spin')
detected2=nx.get_node_attributes(GTot1,'detected')
viralload2=nx.get_node_attributes(GTot1,'viral load')
timerecovered2=nx.get_node_attributes(GTot1,'init time recovered')
timeinfected2=nx.get_node_attributes(GTot1,'init time infected')
timesick2=nx.get_node_attributes(GTot1,'init time sick')
socialtemp2=nx.get_node_attributes(GTot1,'social temp')

if resuming==0:
    with open(outfilename+"/Nodes_sume_"+str(sume)+"_tsocdist_"+str(tsocdist)+"_tsocdistlift_"+str(tsocdistlift)+"_mobility_"+str(mobility)+"_resuming_"+str(resuming)+"_runn_"+str(runn)+".csv","w") as out_file_Graph:
        for i in GTot1.nodes():
            out_string_Graph=""
            out_string_Graph+=str(i)
            out_string_Graph+=","+str(gender2[i])+","+str(age2[i])+","+str(tausick2[i])+","+str(taurecovered2[i])+","+str(tauinfected2[i])
            out_string_Graph+="\n"      
            out_file_Graph.write(out_string_Graph)
    
    out_file_Graph.close()
        
    with open(outfilename+"/Edges_sume_"+str(sume)+"_tsocdist_"+str(tsocdist)+"_tsocdistlift_"+str(tsocdistlift)+"_mobility_"+str(mobility)+"_resuming_"+str(resuming)+"_runn_"+str(runn)+".csv","wb") as out_file_Graph_edge:
        nx.write_edgelist(GTot1,out_file_Graph_edge,data=True)
    
    
    out_file_Graph_edge.close()

    
    
    with open(outfilename+"/init_edge_sume_"+str(sume)+"_tsocdist_"+str(tsocdist)+"_tsocdistlift_"+str(tsocdistlift)+"_mobility_"+str(mobility)+"_resuming_"+str(resuming)+"_runn_"+str(runn)+".csv","wb") as out_file_init_Graph_edge:
        nx.write_edgelist(GTot1,out_file_init_Graph_edge,data=True)
    
    out_file_init_Graph_edge.close()
    
GTot4=GTot1.copy()
lifting=0
reloaded=0
bt=0
pmobility=pmobility1

#gathering each group members
hospitalized1=[v for v in GTot1.node if GTot1.node[v]['state'] == 'hospitalized']
recovered1=[v for v in GTot1.node if GTot1.node[v]['state'] == 'recovered']
infected1=[v for v in GTot1.node if GTot1.node[v]['state'] == 'infected']
susceptible1=[v for v in GTot1.node if GTot.node[v]['state'] == 'susceptible']
other1=[v for v in GTot1.node if GTot.node[v]['state'] == 'other']

measures=0
timevec=[0]

checktime2=0
contador_dias = 0

with open(outfilename+"/R0_sume_"+str(sume)+"_tsocdist_"+str(tsocdist)+"_tsocdistlift_"+str(tsocdistlift)+"_mobility_"+str(mobility)+"_resuming_"+str(resuming)+"_runn_"+str(runn)+".csv","a") as out_file_R0:
    writerR0=csv.writer(out_file_R0, delimiter=",",lineterminator="\n")

    with open(outfilename+"/Sim_sume_"+str(sume)+"_tsocdist_"+str(tsocdist)+"_tsocdistlift_"+str(tsocdistlift)+"_mobility_"+str(mobility)+"_resuming_"+str(resuming)+"_runn_"+str(runn)+".csv","a",newline="") as out_file_Sim:
        writer=csv.writer(out_file_Sim, delimiter=',',lineterminator='\n')
    
        while (check==0) and (t<100*dayduration):  # and contador_dias<=15:
            #out_file_Sim=open(outfilename+"/Sim_sume_"+str(sume)+"_tsocdist_"+str(tsocdist)+"_tsocdistlift_"+str(tsocdistlift)+"_mobility_"+str(mobility)+"_resuming_"+str(resuming)+"_runn_"+str(runn)+".csv","w+")
        
            lambdacontagion=[]
            lambdas=[]
            proctyp=[]
            changedperson=[]
            lambdasick=[]
            lambdarecovered=[]
            lambdainfected=[]
            lambdahosp=[]
            lambdarec=[]
            update=0
                    
            Gtij,spincand,pij=probIJ(GTot1,betafree)            
        
            #print ('pij ', pij)
            
            sumi,proctyp,changedperson,totproc,tottim=probNM(GTot1,pij,spincand,dayduration,t,alphasick,betasick,alpharecovered,betarecovered,alphainfected,betainfected,alphahosp,betahosp,alpharec,betarec,hospitalized1,infected1,recovered1,susceptible1,other1,cluster)
            #print(totproc)
            checktime=0
            t=t+tottim
#            proctodo=np.random.choice(len(totproc),1,p=totproc)
#            proctyptodo=proctyp[int(proctodo)]
#            i=changedperson[int(proctodo)]
            if (math.isclose(sum(totproc), 1, abs_tol=10**-accur)==True):
                proctodo=np.random.choice(len(totproc),1,p=totproc)
                proctyptodo=proctyp[int(proctodo)]
                i=changedperson[int(proctodo)]
            else:
                if cluster==0:
                    print('sum of probabilities: ',sum(totproc))
                proctyptodo=6
                checktime2=1
                out_string_Sim=[]
                out_string_Sim.append(str(t))
                out_string_Sim.append('exited with error')
                writer.writerow(out_string_Sim)
                
            if cluster==0:
                print('new time: ', t, " in days: ", t/250)
                print('type of update: ', int(proctyptodo))
            
            
            
            if (resuming>0) and (reloaded==0):
                reloaded=1
                GTot1,GTot4=reloading(sume,tsocdist,tsocdistlift,mobility,resuming,runn,cluster,outfilename)
                
                
            #out_string_Sim=""
            if (t>tsocdist) and (lifting==0):
                bt=1*3
            if (t>tsocdist) and (measures==0):
                measures=1
                if cluster==0:
                    print('social distancing and quarantine are applied at time: ',t)
                quarantine=1
                pmobility=pmobilitysocdist
                
                GTot1,Go,Gpop,out_string_Sim1=social_dist(t,Gpop,psocdist,GTot1,Go,posocdist,quarantine,cluster)
                out_string_Sim=out_string_Sim1
                                            
                                
            stilldetected=0        
            if (t>tsocdistlift) and (lifting==0):
                GTot1,pmobility,stilldetected,quarantine,bt,lifting,out_string_Sim1=social_dist_lifting(t,GTot1,GTot4,pmobilityafter,cluster,pmobility,quarantine,bt)
                out_string_Sim=out_string_Sim1
                
            
            
            if proctyptodo==0:
                a=i
                out_string_Sim1,GTot1,update,newinfdet,newinf=proc_infection(spincand,pij,GTot1,quarantine,pd,newinfdet,a,t,newinf,R01,Gtij,cluster)
                out_string_Sim=out_string_Sim1
            
            
            
            
            if (proctyptodo==1):
        
                update=1
                GTot1,out_string_Sim1=proc_rec_hosp(GTot1,i,t,newrec,quarantine,newrecdet,cluster)
                out_string_Sim=out_string_Sim1
        
            #ss=i
            if (proctyptodo==2):
                out_string_Sim=[]
                update=1
                if cluster==0:
                    print('time:',t)
                    print(i,'died')
                out_string_Sim.append(str(t))
                #out_string_Sim+=","+str(i)+" died"
                out_string_Sim.append(str(i)+" died")
                #out_string_Sim+="\n"
                numdead+=1
                newdead+=1
                GTot1.remove_node(i)
                Gpopu.remove_node(i)
            
            if (proctyptodo==3):
                out_string_Sim=[]
                update=1
                if cluster==0:
                    print('time:',t)
                    print(i,'not immune anymore')
                out_string_Sim.append(str(t))
                GTot1.node[i]['state']='susceptible'
                #out_string_Sim+=","+str(i)+" not immune anymore"
                out_string_Sim.append(str(i)+" not immune anymore")
                #out_string_Sim+="\n"
        
                            
                
            
            if (proctyptodo==4):#hospitalized
                GTot1,update,out_string_Sim1,newhosp=proc_hosp(quarantine,GTot1,newhosp,i,t,cluster)
                out_string_Sim=out_string_Sim1
                
            #ss=i
            
            if (proctyptodo==5):
                GTot1,update,out_string_Sim1,newrec,newrecdet=proc_rec_inf(GTot1,t,i,quarantine,newrecdet,newrec,cluster)
                out_string_Sim=out_string_Sim1
        
            if update==1:
                #print('hola')
                writer.writerow(out_string_Sim)
                
                
            
            if (update==1) and (proctyptodo==0):
                nb=Gtij.neighbors(a)
                for j in nb:
                    #out_string_R0=""
                    out_string_R0=[]
                    #out_string_R0+=str(t)
                    out_string_R0.append(str(t))
                    #out_string_R0+=","+str(j)
                    out_string_R0.append(str(j))
                    #out_string_R0+="\n"
                    
                    #out_file_R0.write(out_string_R0)
                    writerR0.writerow(out_string_R0)
            
            
            
            GTot1=plotting(GTot1)
        
                    
        #find time of stopping of disease
            
            for q in GTot1.nodes():
                if (((GTot1.node[q]).get('state')).find('h')==0) or (((GTot1.node[q]).get('state')).find('i')==0):
                    checktime=1
            
            if ((checktime==0) and (check==0)) or (checktime2==1):
                stoptime=t
                check=1
            
            if cluster==0:     
            #plotting network at time t   
                if update==1:
                    fig3 = plt.figure(figsize=(20,12))
                    plt.title('Network of infected, etc',size=30)
                    nx.draw_networkx_nodes(GTot1,pos,node_size=[importance for importance in nx.get_node_attributes(GTot1,'importance').values()],node_color=[color for color in nx.get_node_attributes(GTot1,'color').values()],node_shape='o')
                    nx.draw_networkx_labels(GTot1,pos)
                    nx.draw_networkx_edges(GTot1,pos,alpha=0.5,width=[size for size in nx.get_edge_attributes(GTot1,'size').values()],edge_color=[color for color in nx.get_edge_attributes(GTot1,'color').values()])
                    plt.show()
                    time.sleep(1)
                    plt.clf()
                    plt.cla()
                    plt.close()
            
        #gathering all the groups for the new step   
            hospitalized1=[v for v in GTot1.node if GTot1.node[v]['state'] == 'hospitalized']
            recovered1=[v for v in GTot1.node if GTot1.node[v]['state'] == 'recovered']
            infected1=[v for v in GTot1.node if GTot1.node[v]['state'] == 'infected']
            infecteddetected=[]
            recovereddetected=[]
            for v in GTot1.nodes():
                if (GTot1.node[v]['detected'] == 'yes'):
                    if (GTot1.node[v]['state'] == 'infected'):
                        infecteddetected.append(v)
                    if (GTot1.node[v]['state'] == 'recovered'):
                        recovereddetected.append(v)
                    
            susceptible1=[v for v in GTot1.node if GTot1.node[v]['state'] == 'susceptible']
            other1=[v for v in GTot1.node if GTot1.node[v]['state'] == 'other']
        
        #find maximum hospitalized
            if len(hospitalized1)>numhosmax:
                numhosmax=len(hospitalized1)
            
            totpop=len(hospitalized1)+len(recovered1)+len(infected1)+len(susceptible1)
        
        #new and total cases
            newinfected.append(newinf)
            sum1=0
            for q in newinfected:
                sum1+=q
            totinfected.append(sum1)
            newinf=0
            dailyinf.append(len(infected1))
            
            #if quarantine==1:
            newinfecteddetected.append(newinfdet)
            sum1=0
            for q in newinfecteddetected:
                sum1+=q
            totinfecteddetected.append(sum1)
            newinfdet=0
            dailyinfdet.append(len(infecteddetected))
            
            newdead1.append(newdead)
            sum1=0
            for q in newdead1:
                sum1+=q
            totdead.append(sum1)
            newdead=0
            
            newrecovered.append(newrec)
            sum1=0
            for q in newrecovered:
                sum1+=q
            totrec.append(sum1)
            newrec=0
            dailyrec.append(len(recovered1))
            
            #if quarantine==1:
            newrecovereddetected.append(newrecdet)
            sum1=0
            for q in newrecovereddetected:
                sum1+=q
            totrecovereddetected.append(sum1)
            newrecdet=0
            dailyrecdet.append(len(recovereddetected))
            
            newhospitalized.append(newhosp)
            sum1=0
            for q in newhospitalized:
                sum1+=q
            tothosp.append(sum1)
            newhosp=0
            dailyhosp.append(len(hospitalized1))
            
#            contador_dias += 1
#            print("Llevamos: ", contador_dias, " dias")
            
            
            
        #saving current network
            R02=nx.get_node_attributes(GTot1,'R0')
            tausick2=nx.get_node_attributes(GTot1,'tausick')
            taurecovered2=nx.get_node_attributes(GTot1,'taurecovered')
            tauinfected2=nx.get_node_attributes(GTot1,'tauinfected')
            gender2=nx.get_node_attributes(GTot1,'gender')
            age2=nx.get_node_attributes(GTot1,'age')
            state2=nx.get_node_attributes(GTot1,'state')
            spin2=nx.get_node_attributes(GTot1,'spin')
            detected2=nx.get_node_attributes(GTot1,'detected')
            viralload2=nx.get_node_attributes(GTot1,'viral load')
            timerecovered2=nx.get_node_attributes(GTot1,'init time recovered')
            timeinfected2=nx.get_node_attributes(GTot1,'init time infected')
            timesick2=nx.get_node_attributes(GTot1,'init time sick')
            socialtemp2=nx.get_node_attributes(GTot1,'social temp')
            numofneigh2=nx.get_node_attributes(GTot1,'num of neigh')
            numofneighdet2=nx.get_node_attributes(GTot1,'num of neigh det')
            maxinumneigh=0
            for y in numofneigh2:
                if numofneigh2[y]>maxinumneigh:
                    maxinumneigh=numofneigh2[y]
            for y in numofneighdet2:
                if numofneighdet2[y]>maxinumneigh:
                    maxinumneigh=numofneigh2[y]   
            
            
            with open(outfilename+"/Current_sume_"+str(sume)+"_tsocdist_"+str(tsocdist)+"_tsocdistlift_"+str(tsocdistlift)+"_mobility_"+str(mobility)+"_resuming_"+str(resuming)+"_runn_"+str(runn)+".csv","w") as out_file_Current_Graph:
                for q in GTot1.nodes():
                    out_string_Current_Graph=""
                    out_string_Current_Graph+=str(t)+","+str(q)
                    if q.find('o')==0:
                        out_string_Current_Graph+=","+str(0)+","+str(0)+","+str(0)+","+str(0)+","+str(viralload2[q])+","+str(detected2[q])+","+str(spin2[q])+","+str(state2[q])+","+str(gender2[q])+","+str(age2[q])+","+str(tausick2[q])+","+str(taurecovered2[q])+","+str(tauinfected2[q])+","+str(socialtemp2[q])+","+str(0)+","+str(0)
                        for q1 in range(maxinumneigh):
                            out_string_Current_Graph+=","+str(0)
                    else:
                        out_string_Current_Graph+=","+str(R02[q])+","+str(timesick2[q])+","+str(timeinfected2[q])+","+str(timerecovered2[q])+","+str(viralload2[q])+","+str(detected2[q])+","+str(spin2[q])+","+str(state2[q])+","+str(gender2[q])+","+str(age2[q])+","+str(tausick2[q])+","+str(taurecovered2[q])+","+str(tauinfected2[q])+","+str(0)+","+str(numofneigh2[q])+","+str(numofneighdet2[q])
                        if (numofneigh2[q]>0) or (numofneighdet2[q]>0):
                            if (numofneigh2[q]>0):
                                for q1 in range(numofneigh2[q]):
                                    out_string_Current_Graph+=","+str(GTot1.node[q]['neigh'+str(q1)])
                                for q1 in range(maxinumneigh-numofneigh2[q]):
                                    out_string_Current_Graph+=","+str(0)
                            if (numofneighdet2[q]>0):
                                for q1 in range(numofneighdet2[q]):
                                    out_string_Current_Graph+=","+str(GTot1.node[q]['neigh det'+str(q1)])
                                for q1 in range(maxinumneigh-numofneighdet2[q]):
                                    out_string_Current_Graph+=","+str(0)
                        else:
                            for q1 in range(maxinumneigh):
                                out_string_Current_Graph+=","+str(0)
                    out_string_Current_Graph+="\n"
              
                    out_file_Current_Graph.write(out_string_Current_Graph)
        
        
            out_file_Current_Graph.close()                            
                                    
            with open(outfilename+"/Current_sume_"+str(sume)+"_tsocdist_"+str(tsocdist)+"_tsocdistlift_"+str(tsocdistlift)+"_mobility_"+str(mobility)+"_resuming_"+str(resuming)+"_runn_"+str(runn)+".csv","wb") as out_file_Current_Graph_edge:
                nx.write_edgelist(GTot1,out_file_Current_Graph_edge,data=True)
        
            out_file_Current_Graph_edge.close()
            
            
            
            #resetting times of interactions for the new step
            GTot1,Go,Gpop,betafree=next_step(dayduration,t,GTot1,Gpop,Go,Gr1,Gr2,Gr3,Gr4,Gr5,Gromerged,GSW,nlistmerged,quarantine,bt,alphatemp,betatemp,mobility,pmobility,mobfreq,N1,N2,N3,N4,N5,posocdist,po,p,psocdist,hospitalized1,infected1,recovered1,susceptible1,other1,betafree)
                    
            timevec.append(t)                        
            #t+=1

out_file_Sim.close()
out_file_R0.close()

tiempo_final = tmm() 
tiempo_ejecucion = tiempo_final - tiempo_inicial
print('El tiempo de ejecucion en segundos fue:',tiempo_ejecucion) #En segundos
print("La simulacion contiene: " + str(t/dayduration) + " dias")
lista_datos = []
lista_datos.append(tiempo_ejecucion)
lista_datos.append(t/dayduration)
np.savetxt(outfilename+"/datos_" + str(runn) + '.txt', lista_datos)
#if cluster==0:
#    if (stoptime>tsocdist):
#        print('social distancing measures were implemented at day',tsocdist//dayduration)
#    print('lifting', lifting)
#    print('percentage of dead people to total:',numdead/(N-No)) 
#    print('disease disappeared at time',stoptime//dayduration)
#    print('maximum number in hospital',numhosmax)
#    print('percentage of dead people to contagiated',numdead/(1+totinfected[len(totinfected)-1]))
#    print('percentage of recovered in total',totrec[len(totrec)-1]/(1+totinfected[len(totinfected)-1]))
#    if quarantine==1:    
#        print('percentage of recovered of those detected',totrecovereddetected[len(totrecovereddetected)-1]/(1+totinfecteddetected[len(totinfecteddetected)-1]))
#    print('currently ',len(recovered1)/totpop,'of the population has immunity')
#
#
#    fig4 = plt.figure(figsize=(12.5,7.5))
#    axe4=fig4.gca()
#    #timeaxis=range(stoptime+2)
#    timeaxis=timevec
#        
#    
#    plotnewdead,=plt.plot(timeaxis,newdead1)
#    plotnewinf,=plt.plot(timeaxis,newinfected)
#    
#    
#    
#    axe4.set_xlabel('time',fontsize=30)
#    
#    
#    plt.legend([plotnewdead,plotnewinf], ['new dead','new infected'],fontsize=20)
#    
#    fig5 = plt.figure(figsize=(12.5,7.5))
#    axe5=fig5.gca()
#    #timeaxis=range(stoptime+2)
#    timeaxis=timevec
#    
#    plotdailyhosp,=plt.plot(timeaxis,dailyhosp)
#    
#    axe5.set_xlabel('time',fontsize=30)
#    
#    
#    plt.legend([plotdailyhosp], ['daily hospitalized'],fontsize=20)
#    
#    
#    fig7 = plt.figure(figsize=(12.5,7.5))
#    axe7=fig7.gca()
#    
#    
#    yt=0
#    #totdays=stoptime//dayduration
#    totdays=timevec[len(timevec)-1]//dayduration
#    
#    stoptime=math.floor(timevec[len(timevec)-1])
#    
#    ytd=1
#    newinfectedD=[0, 0]
#    newinfected1=0
#    while yt<stoptime+2:
#        if (yt//dayduration>(ytd-1)):
#    
#            ytd+=1
#    
#            newinfected1=0
#            newinfectedD.append(newinfected1)
#        else:
#            newinfected1=newinfected1+newinfected[yt]
#        newinfectedD[ytd]=newinfected1
#        yt+=1
#    
#    
#    timeaxisD=range(totdays+2)
#    
#    axe7.set_xlabel('time',fontsize=30)
#    
#    plotnewinfD,=plt.plot(timeaxisD,newinfectedD)
#    plt.legend([plotnewinfD], ['new infected daily'],fontsize=20)

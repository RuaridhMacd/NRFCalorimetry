# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 21:40:42 2016

@author: Ruaridh Macdonald
"""

import NRFCal1D

import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

import time # For timing parts of the code. Feel free to remove / comment out

from scipy.optimize import fsolve

# ------ ------ ------
# Import arguments and check for errors
# ------ ------ ------
parser = argparse.ArgumentParser(description='Models the attenuation of a Bremsstrahlung beam through a 1D object and rotating ''foil''. The object and foil are described in a material input file.\nExample:\
    .......................................................................\
    $./NRFCal1D_chopper_run.py -matList=matList.txt -totalTime=10.0 -beamFlux=1e12  \
    .......................................................................\
    Produces plots of the Bremsstahlung energy profile, cross sections used and the time varying calorimeter signal. Assumes perfect detector efficiency at all energies.')
        
# -h and --help options exist by default
parser.add_argument('-matList', help='File name of list of isotopes and their properties \n Format: A Z numDensity frequency', type=str)

parser.add_argument('-energyBins', help='Number of energy bins (log-spaced), default = 1000 ', type=float, default=1e3)
parser.add_argument('-energyMin', help='Minimum energy (MeV), default = 0.001 ', type=float, default=1e-3)
parser.add_argument('-energyMax', help='Maximum energy (MeV), default = 10', type=float, default=10)

parser.add_argument('-beamFlux', help='Source fluence per cycle, default = 1e10 ', type=float, default=1e10)
parser.add_argument('-beamFreq', help='Source frequncy (Hz), default = 1', type=float, default=1.0)
parser.add_argument('-beamDutyFactor', help='Source duty factor fraction, default = 1.0 ', type=float, default=1.0)
parser.add_argument('-beamNoise', help='stan dev. / mean of source noise as Gaussian r.v., default = 0.01 ', type=float, default=0.01)

parser.add_argument('-totalTime', help='Total beam time (seconds), default = 1.0', type=float, default=1.0)
parser.add_argument('-timeStep', help='Time step (seconds), default = 0.001', type=float, default=0.001)

parser.add_argument('-chopperProfile', help='Choose the foil profile as it changes with time. \nOptions are ''sqr'' or ''sin'', default = sqr ', type=str, default='sqr')
parser.add_argument('-idealFiller', help='Choose the foil profile as it changes with time. \nOptions are ''sqr'' or ''sin'', default = sqr ', type=float, default=1.0)

parser.add_argument('-plotOn', help='Plot results? Yes = 1, No = 0, default = 1', type=int, default=1)
parser.add_argument('-figNum', help='Figure number to plot to, defauly = 1',type=int, default=1)

args = parser.parse_args()

if args.matList == None: sys.exit("User must specify a material file describing the foil and warhead isotopic content")
if args.energyBins < 2: 
    sys.exit("Must discretize energy into more than one bin")
else: numBins = args.energyBins
if args.energyMax<=args.energyMin: sys.exit("\nMaximum energy must be greater than minimum detector energy and positive")
if args.energyMin<=0 or args.energyMax<=0: sys.exit("\nEnergy limits must be positive")    
if args.beamFlux<=0: sys.exit("\nBeam flux must be positive")
if args.beamFreq<=0: sys.exit("\nBeam frequency must be positive")
if args.beamDutyFactor<=0 or args.beamDutyFactor>1.0: sys.exit("\nBeam duty factor must be between 0 and 1")
if args.totalTime<=0: sys.exit("\nTotal beam time must be positive")
if args.timeStep<=0: sys.exit("\nNumber of time steps must be positive")
if args.chopperProfile != 'sqr' and args.chopperProfile != 'sin': sys.exit("\nUnrecognised chopperProfile. Use 'sin' or 'sqr'")
    
numBins = args.energyBins           # Number of energy bins - These will be log-spaced between the max and min
energyMin = args.energyMin           # All energies are in MeV
energyMax = args.energyMax
plotOn = args.plotOn
maxTime = int(args.totalTime/args.timeStep)-1

guessList = []
guessList_error = []
for reps in range(24):
    t0 = time.time()
    
    NRFDatabase = open('standalone_extended.dat','r') 
    
    # Non-resonant (NR) data
    NRData = np.loadtxt('nonResonantAttenuation.txt',dtype=float,delimiter='|') # This file has to be carefully formatted
    
    t1 = time.time()
    # ------ ------ ------
    # Run main file
    # This is in one function to assist with repeat runs
    # ------ ------ ------   
    calorimeterOut, calDiffOut, matDetails, energyBins, emitList, crossSections, sourceDetails, timeDetails, testObjectList = NRFCal1D.chopper_run(args.matList,NRFDatabase,NRData,numBins,args.energyMin,args.energyMax,args.beamFlux,args.beamFreq,args.beamDutyFactor,args.beamNoise,args.totalTime,args.timeStep,args.chopperProfile,args.idealFiller)
     
    t2 = time.time()
    # The outputs are grouped in lists:
    # calorimeterOut = [calorimeter,  calorimeterMean]
    # calDiffOut = [calDiff, calDiffMean]
    # matDetails = [matList, nDensList, thickList, freqList]
    # sourceDetails = [bremsFlux,sourceStatus,sourceStrength]
    # crossSections = [crossSection_micro_nonRes, crossSection_micro_NRF, crossSection_macro_nonRes, crossSection_macro_NRF]
    # timeDetails = [maxTime, timeSeries, sourceStatus, bremsFluxTime]
    
    NRFDatabase.close()
    
    [matList, nDensList, thickList, freqList] = matDetails
    nDensList = np.array(nDensList)
    thickList = np.array(thickList)
        
    [calDiff, calDiffMean] = calDiffOut
    [crossSection_micro_nonRes, crossSection_micro_NRF] = crossSections 
    
    NRData = np.split(NRData,[1],axis=1)
    NREnergy = NRData[0] # Vector of energies that the database uses
    NRData = NRData[1]   # Non-resonant attenuation data by isotope, z = 1:100
       
    bremsFlux = sourceDetails[0]
    
    # ------ ------ ------
    # Spectral analysis of calorimeter signal
    # ------ ------ ------
    #temp = 2**np.ceil(np.log2(np.max(freqList)))
    xT = np.fft.fftfreq(maxTime+1,args.totalTime/(maxTime+1))
    yT = np.fft.fft(calDiff)/( (maxTime+1)/4 )
    mT = np.fft.fft(calDiffMean)/( (maxTime+1)/4 )
    
    t3 = time.time()
    
    print('Load data: ',t1-t0,'\nMain loop: ',t2-t1,'\nFFT: ',t3-t2)
    # ------ ------ ------
    # Plot results
    # ------ ------ ------
    if plotOn == 1:
        NRFCal1D.chopper_plot(args.figNum,matList,energyBins,freqList,calorimeterOut,calDiffOut,xT,yT,mT,sourceDetails,timeDetails,testObjectList)
        
    # ------ ------ ------
    # Perform a reverse calculation to see the maximum possible 
    # inference accuracy we could achieve given the number of
    # counts actually measured (i.e. with randomness)
    # ------ ------ ------
    freqList_unique = np.unique(np.array(freqList))
        
    def fourierCheck(_testObject, _energyBins, _testObjectList, _bremsFlux):
        # This function calculates the result of the Fourier transform at the frequencies of interest using the actual data
        # We can then use this result to estimate the maximum possible achievable accuracy
    
        testObject = _testObject
        testObjectList = _testObjectList
        energyBins = _energyBins
        bremsFlux = _bremsFlux
        
        # Make a list of all of the NRF resonance energies for us to calculate against
        energyIndex = np.squeeze([testObject.crossSection_moi_macro_NRF > 0.0])
        energyList = energyBins[energyIndex]
        
        itemTerm = np.ones(len(energyList))
        for item in testObjectList:
            if item.freq == testObject.freq: # Active object will be included later
                continue
            elif item.freq == 0.0:
                objectTerm = np.exp( -np.sum( (item.crossSection_moi_macro_nonRes[energyIndex,:] + item.crossSection_moi_macro_NRF[energyIndex,:]), axis=1) )
            else:
                itemTerm = np.multiply(itemTerm, np.exp( -np.sum( item.crossSection_moi_macro_nonRes[energyIndex], axis=1 ) ) ) 
                
        energyTerm = energyList*bremsFlux[energyIndex][:,0]*args.beamDutyFactor*1e-5
        foilNRTerm = np.exp( -np.sum( testObject.crossSection_moi_macro_nonRes[energyIndex], axis=1 ) )
        foilNRFTerm = ( 1.0-np.exp(-np.sum( testObject.crossSection_moi_macro_NRF[energyIndex], axis=1 )) )
    
    #    print(np.shape(energyTerm))
    #    print(np.shape(foilNRTerm))
    #    print(np.shape(foilNRFTerm))
        
        fourierCoeff = itemTerm*objectTerm*energyTerm*foilNRTerm*foilNRFTerm*2.0/np.pi # Store coefficients for calculations - [other,Object]
    
        return np.array(fourierCoeff), energyList
        
    freqList = np.array(freqList)
            
    for testObject in testObjectList:
        if testObject.freq == 0.0:
            continue    
        else:
            fourierCoeff, energyList = fourierCheck(testObject,energyBins,testObjectList,bremsFlux)
            testObject.fCheck = fourierCoeff
    
    # ------ ------ ------
    # Infer object properties
    # ------ ------ ------
    
    def calcMCoeffs(_testObject,_energyBins,_testObjectList,_bremsFlux,_matList_unique):
        
        testObject = _testObject
        testObjectList = _testObjectList
        energyBins = _energyBins
        bremsFlux = _bremsFlux
        
        # Make a list of all of the NRF resonance energies for us to calculate against
        energyIndex = np.squeeze([testObject.crossSection_moi_macro_NRF > 0.0])
        energyList = energyBins[energyIndex]
        
        varCoeff = np.ones([len(energyList),len(_matList_unique)])
        
        itemTerm = np.ones(len(energyList))
        for item in testObjectList:
            if item.freq == testObject.freq: 
                continue
            elif item.freq == 0.0:
                counter = -1
                for mat in item.matList:
                    mat = list(mat)
                    counter += 1
                    varCoeff[:,counter] = crossSection_micro_nonRes[energyIndex,_matList_unique.index(mat)]+crossSection_micro_NRF[energyIndex,_matList_unique.index(mat)]
            else:
                itemTerm = np.multiply(itemTerm, np.exp( -np.sum( item.crossSection_moi_macro_nonRes[energyIndex], axis=1 ) ) ) 
                
        energyTerm = energyList*bremsFlux[energyIndex][:,0]*args.beamDutyFactor*1e-5
        foilNRTerm = np.exp( -np.sum( testObject.crossSection_moi_macro_nonRes[energyIndex], axis=1 ) )
        foilNRFTerm = ( 1.0-np.exp(-np.sum( testObject.crossSection_moi_macro_NRF[energyIndex], axis=1 )) )
        
        frontCoeff = itemTerm*energyTerm*foilNRTerm*foilNRFTerm*2.0/np.pi
                 
        return frontCoeff,varCoeff
    
    def func2Solve(linDensGuess,yTPeakList,frontCoeffList,varCoeffList):
        
        solution = np.ones(np.shape(linDensGuess)) 
        yTPeakList = np.array(yTPeakList)
        
        for i in range(len(linDensGuess)):
            
            solution[i] = yTPeakList[i] - np.sum( np.multiply( frontCoeffList[i],np.exp( -np.sum( np.multiply(np.abs(np.array(linDensGuess)) , varCoeffList[i]) ,1) ) ) ,0)
                 
        return tuple(solution)
    
    matList_unique = [list(x) for x in set(tuple(x) for x in matList)]
    
    frontCoeffList = []
    varCoeffList = []
    yTPeakList = []
    mTPeakList = []
    for testObject in testObjectList:
        if testObject.freq == 0.0:
            continue
        else:
            
            frontCoeff,varCoeff = calcMCoeffs(testObject, energyBins, testObjectList, bremsFlux, matList_unique)
            frontCoeffList.append(frontCoeff)
            varCoeffList.append(varCoeff)
            
            temp = np.where(xT==testObject.freq)
    #        yTPeakList.append( np.abs(yT[temp]) )
    #        mTPeakList.append( np.abs(mT[temp]) )
            yTPeakList.append( np.abs(np.imag(yT[temp])) )
            mTPeakList.append( np.abs(np.imag(mT[temp])) )
            #    yTPeakList.append( np.imag(yT[temp]) )
            #    mTPeakList.append( np.imag(mT[temp]) )
        
    linDensGuess = 0.03*np.ones([1,len(testObjectList)-1])
    linDensGuessMean = 0.03*np.ones([1,len(testObjectList)-1])
    
    linDensGuess = fsolve( func2Solve,tuple(linDensGuess),args=(yTPeakList,frontCoeffList,varCoeffList) )
    linDensGuessMean = fsolve( func2Solve,tuple(linDensGuessMean),args=(mTPeakList,frontCoeffList,varCoeffList) )
    
    guessList.append(linDensGuess)
    guessList_error.append(np.divide(linDensGuess,nDensList[np.array(freqList)==0.0]*thickList[np.array(freqList)==0.0]))

#linDensGuessMean = linDensGuessMean.T


plt.figure(args.figNum+3)
plt.clf()
plt.subplot(311)
plt.plot(xT[0:int((maxTime+1)/2)],np.abs(yT[0:int((maxTime+1)/2)]),'k')
plt.subplot(312)
plt.plot(xT[0:int((maxTime+1)/2)],np.real(yT[0:int((maxTime+1)/2)]),'k')
plt.subplot(313)
plt.plot(xT[0:int((maxTime+1)/2)],np.imag(yT[0:int((maxTime+1)/2)]),'k')

counter = -1
for testObject in testObjectList:
    if testObject.freq == 0.0:
        warhead = testObject
    else:
        freq = testObject.freq
        counter += 1
        
        print("\n------ Material with ",freq,"Hz chopper frequency -----")
        print("Linear density estimate: ",linDensGuess[counter])
        print("estimate / true: ",linDensGuess[counter]/warhead.nDensList[counter-1]/warhead.thickList[counter-1])
        
        print("\nEstimate w/ source noise only: ",linDensGuessMean[counter])
        print("estimate / true: ",linDensGuessMean[counter]/warhead.nDensList[counter-1]/warhead.thickList[counter-1])
        
        temp = np.where(xT==freq)
        print("\nMax possible accuracy:")
        print("Abs ratio = ",np.abs(yT[temp])/np.sum(testObject.fCheck))
        print("Real ratio = ",np.real(yT[temp])/np.sum(testObject.fCheck))
        print("Imag ratio = ",np.imag(yT[temp])/np.sum(testObject.fCheck))
        
           
        plt.figure(args.figNum+3)
        plt.subplot(311)
        plt.plot(freq,np.sum(testObject.fCheck),'ro')
        plt.subplot(312)
        plt.plot(freq,np.sum(testObject.fCheck),'ro')
        plt.subplot(313)
        plt.plot(freq,np.sum(testObject.fCheck),'ro')
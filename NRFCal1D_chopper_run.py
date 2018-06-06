# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 21:40:42 2016

@author: Ruaridh Macdonald
"""

import NRFCal1D
import NRFCal1D_inference

import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

import time # For timing parts of the code. Feel free to remove / comment out
t0 = time.time()
from scipy.optimize import least_squares

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

# ------ ------ ------
# Extract data from material files
# ------ ------ ------
# NRF data
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
sourceDetails.append(args.beamDutyFactor)

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
    
freqList = np.array(freqList)
        
for testObject in testObjectList:
    if testObject.freq == 0.0:
        continue    
    else:
        fourierCoeff, energyList = NRFCal1D_inference.fourierCheck(testObject,energyBins,testObjectList,sourceDetails)
        testObject.fCheck = fourierCoeff

# ------ ------ ------
# Infer object properties
# ------ ------ ------

matList_unique = [list(x) for x in set(tuple(x) for x in matList)]

frontCoeffList = []
varCoeffList = []
yTPeakList = []
mTPeakList = []
for testObject in testObjectList:
    if testObject.freq == 0.0:
        continue
    else:
        
        frontCoeff,varCoeff = NRFCal1D_inference.calcMCoeffs(testObject, energyBins, testObjectList, matList_unique, crossSections, sourceDetails)
        frontCoeffList.append(frontCoeff)
        varCoeffList.append(varCoeff)
        
        temp = np.where(xT==testObject.freq)
#        yTPeakList.append( np.abs(yT[temp]) )
#        mTPeakList.append( np.abs(mT[temp]) )
        yTPeakList.append( np.abs(np.imag(yT[temp])) )
        mTPeakList.append( np.abs(np.imag(mT[temp])) )
        #    yTPeakList.append( np.imag(yT[temp]) )
        #    mTPeakList.append( np.imag(mT[temp]) )
    
linDensGuess = 0.03*np.ones([len(testObjectList)-1])
linDensGuessMean = 0.03*np.ones([len(testObjectList)-1])

linDensBounds = ()
densBound = (0,None)
for i in range(len(testObjectList)-1):
    linDensBounds = linDensBounds+(densBound,)
    
linDensGuess = least_squares( NRFCal1D_inference.func2Solve,linDensGuess,args=(yTPeakList,frontCoeffList,varCoeffList),bounds=[0, np.inf])
linDensGuessMean = least_squares( NRFCal1D_inference.func2Solve,linDensGuessMean,args=(mTPeakList,frontCoeffList,varCoeffList),bounds=[0, np.inf])

plt.figure(args.figNum+3)
plt.clf()
plt.subplot(311)
plt.plot(xT[0:int((maxTime+1)/2)],np.abs(yT[0:int((maxTime+1)/2)]),'k')
plt.subplot(312)
plt.plot(xT[0:int((maxTime+1)/2)],np.real(yT[0:int((maxTime+1)/2)]),'k')
plt.subplot(313)
plt.plot(xT[0:int((maxTime+1)/2)],np.imag(yT[0:int((maxTime+1)/2)]),'k')

linDensGuess = linDensGuess.x
linDensGuessMean = linDensGuessMean.x

counter = -1
for testObject in testObjectList:
    if testObject.freq == 0.0:
        warhead = testObject
    else:
        freq = testObject.freq
        counter += 1
        
        print("\n------ Material with %4.2f Hz chopper frequency -----" % (freq,))
        print("Linear density estimate: %6.4f" % linDensGuess[counter])
        print("True linear density: %6.4f" % (warhead.nDensList[counter-1]*warhead.thickList[counter-1]) )
        print("estimate / true: %6.4f" % (linDensGuess[counter]/warhead.nDensList[counter-1]/warhead.thickList[counter-1]) )
        
        print("\nEstimate w/ source noise only: %6.4f" % linDensGuessMean[counter])
        print("estimate / true: %6.4f" % (linDensGuessMean[counter]/warhead.nDensList[counter-1]/warhead.thickList[counter-1]) )
        
        temp = np.where(xT==freq)
        print("\nMax possible accuracy:")
        print("Abs ratio = %6.4f" % ( np.abs(yT[temp])/np.sum(testObject.fCheck)) )
        print("Real ratio = %6.4f" % ( np.real(yT[temp])/np.sum(testObject.fCheck)) )
        print("Imag ratio = %6.4f" % ( np.imag(yT[temp])/np.sum(testObject.fCheck)) )
        
           
        plt.figure(args.figNum+3)
        plt.subplot(311)
        plt.plot(freq,np.sum(testObject.fCheck),'ro')
        plt.subplot(312)
        plt.plot(freq,np.sum(testObject.fCheck),'ro')
        plt.subplot(313)
        plt.plot(freq,np.sum(testObject.fCheck),'ro')
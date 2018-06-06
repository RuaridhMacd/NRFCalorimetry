# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 2016

@author: Ruaridh Macdonald
"""
import math
import numpy as np
import sys
import matplotlib.pyplot as plt

class NRFGamma:
    def __init__(self, _z, _a, _Elevel, _Egamma, _Width, _prob, _GSprob, _J0, _Jr, _TDebye, _nDens, _thickness,_counter):
        self.z      = _z
        self.a      = _a
        self.Elevel = _Elevel  # Energy of the resonant level
        self.Egamma = _Egamma  # Energy of the emitted gamma
        self.Width  = _Width   # Width Gamma_r of the resonant level
        self.prob   = _prob    # Branching ratio brj of decay from resonance to final state
        self.GSprob = _GSprob  # Branching ratio b0r of decay from resonance to ground sate
        self.J0     = _J0      
        self.Jr     = _Jr      
        self.TDebye = _TDebye
        self.nDens  = _nDens   # atom / cm^2 * 1e-24
        self.thickness = _thickness # cm
        self.index = _counter

        # Calculate the energy-integrated cross section
        g = (2.0*self.Jr+1)/(2.0*(2.0*self.J0+1))
        hbarc = 197.327e-15 # MeV m
        self.sigmaInt = 1.0e34 * 2.0 * (math.pi)**2 * g * (hbarc/self.Elevel)**2 * self.Width * self.prob * self.GSprob # eV b
        self.sigmaInt = self.sigmaInt/self.prob
        
        # and the Doppler-broadened peak height
        Mc2 = self.a * 931.454 # MeV
        kB = 8.6173e-11 # MeV/K
        self.Delta = self.Elevel * math.sqrt(2*kB*300.0/Mc2) # FIXME this should use Teff
        self.sigmaDmax = 1.0e28 * 2.0 * (math.pi)**(3.0/2.0) * g * (hbarc/self.Elevel)**2 * self.prob * self.GSprob * self.Width / self.Delta # b
        
class testObject:
    def __init__(self, _freq, _matList_moi, _nDensList_moi, _thickList_moi, _idealFiller,  _matList_filler, _nDensList_filler, _thickList_filler, _matList_unique, _crossSection_micro_nonRes, _crossSection_micro_NRF, _timeSeries, _chopperProfile):
        self.freq = _freq
        self.idealFiller = _idealFiller
        
        self.matList = _matList_moi
        self.nDensList = _nDensList_moi
        self.thickList = _thickList_moi
        
        self.matList_filler = _matList_filler
        self.nDensList_filler = _nDensList_filler
        self.thickList_filler = _thickList_filler
        
        self.chopperProfile = _chopperProfile
        
        self.fCheck = 0.0
        
        crossSection_micro_nonRes = _crossSection_micro_nonRes
        crossSection_micro_NRF = _crossSection_micro_NRF
        
        timeSeries = _timeSeries
        
        self.crossSection_moi_macro_nonRes = np.zeros([np.shape(crossSection_micro_nonRes)[0],len(self.matList)])
        self.crossSection_moi_macro_NRF = np.copy(self.crossSection_moi_macro_nonRes)
        counter = -1    
        for mat in self.matList:
            mat = list(mat)
            counter += 1
            self.crossSection_moi_macro_nonRes[:,counter] = np.multiply((self.nDensList[counter] * self.thickList[counter]), crossSection_micro_nonRes[:,_matList_unique.index(mat)] )
            self.crossSection_moi_macro_NRF[:,counter] = np.multiply((self.nDensList[counter] * self.thickList[counter]), crossSection_micro_NRF[:,_matList_unique.index(mat)] )
            
        self.crossSection_filler_macro_nonRes = np.zeros([np.shape(crossSection_micro_nonRes)[0],len(self.matList_filler)])
        self.crossSection_filler_macro_NRF = np.copy(self.crossSection_filler_macro_nonRes)
        counter = -1    
        if self.idealFiller == 1:
            for mat in self.matList_filler:
                mat = list(mat)
                counter += 1
                self.crossSection_filler_macro_nonRes[:,counter] = np.multiply((self.nDensList_filler[counter] * self.thickList_filler[counter]), crossSection_micro_nonRes[:,_matList_unique.index(mat)] )
        else:            
            for mat in self.matList_filler:
                print("non ideal Filler used")
                mat = list(mat)
                counter += 1
                self.crossSection_filler_macro_nonRes[:,counter] = np.multiply((self.nDensList_filler[counter] * self.thickList_filler[counter]), crossSection_micro_nonRes[:,_matList_unique.index(mat)] )
                self.crossSection_filler_macro_NRF[:,counter] = np.multiply((self.nDensList_filler[counter] * self.thickList_filler[counter]), crossSection_micro_NRF[:,_matList_unique.index(mat)] )
                   
        self.chopperStatus = np.ones(np.shape(timeSeries))
        self.fillerStatus = np.zeros(np.shape(timeSeries))
        if self.freq == 0.0: pass # Test object / warhead
            
        elif self.freq > 0.0: # Chopper materials
            if self.chopperProfile == 'sqr':
                self.chopperStatus = np.round( (1.0 + np.sin(self.freq*2.0*np.pi * timeSeries) )/2 )
                                
            elif self.chopperProfile == 'sin':
                self.chopperStatus = ( (1.0 + np.sin(self.freq*2.0*np.pi * timeSeries) )/2 )
            
            self.fillerStatus = 1.0 - self.chopperStatus
            
        else :
            sys.exit("\freqList Error: a frequency is not formatted correctly")
        
        self.crossSection_inTime_macro_total = np.matrix(np.sum(np.concatenate([self.crossSection_moi_macro_nonRes,self.crossSection_moi_macro_NRF],axis=1),axis=1)).T*np.matrix(self.chopperStatus) + np.matrix(np.sum(np.concatenate([self.crossSection_filler_macro_nonRes,self.crossSection_filler_macro_NRF],axis=1),axis=1)).T*np.matrix(self.fillerStatus)  
        
   
def parse_materials(fileName): 
    if type(fileName) != str:
        print('parse_material ERROR: \nMaterial list file name must be a string')
        return
        
    matList = []   # List of atomic and mass numbers for materials in object
    nDensList = [] # (atom / cm**3) * A * 1e-24 
    thickList = [] # [warhead,foil] thicknesses (cm)
    freqList = []  # Frequency of chopper wheel of that material
    
    matFile = open(fileName,'r')

    for line in matFile:
        line = line.strip()
        words = line.split(' ')
        
        if words[0][0]=="#": continue # Ignore comment lines in the material description
        
        matList.append([int(words[0]),int(words[1])])
        nDensList.append(float(words[2]))
        thickList.append(float(words[3]))
        freqList.append(float(words[4]))
        
    matFile.close()
    
    return matList, nDensList, thickList, freqList
    
def build_source(energyMax,energyBins,beamFlux=1e10,beamFreq=10.0,beamDutyFactor=1.0,beamNoise=0.01,totalTime=1.0,timeStep=0.001):
    maxTime = round(totalTime/timeStep)-1 # Number of time steps to be used, shifted to 0-index

    # Build Bremsstrahlung source using Kramers law approximation
    # Kramers law: Intensity(E) = K/(2*PI*c) * (Emax/E - 1)
    bremsFlux = np.divide(energyMax,energyBins) - 1     
    bremsFlux = bremsFlux[:,None]
    
    sourceStrength = beamFlux/beamDutyFactor

    # Normalize beam strength and multiply by user defined fluence
    temp1 = np.trapz(bremsFlux[:,0],x=energyBins)      
    bremsFlux = np.divide(bremsFlux,np.abs(temp1))
    bremsFlux = np.multiply(bremsFlux,sourceStrength)
    
    sourcePattern = [1]*round(beamDutyFactor/beamFreq/timeStep) + [0]*round((1.0-beamDutyFactor)/beamFreq/timeStep) 
    
    sourceStatus = np.tile(sourcePattern,np.ceil( (maxTime+1)/np.size(sourcePattern)) ) # Repeat source unit/cycle pattern. This may produce extra time steps
    sourceStatus = sourceStatus[0:maxTime+1]  
    perfectFlux = np.matrix(bremsFlux)*np.matrix(sourceStatus)
    
    if beamNoise == 0.0: 
        # No noise added to the source signal
        bremsFluxTime = np.copy(perfectFlux)
    else:
        # Add pulse to pulse noise, Gaussian with sigma/mu = 'beamNoise'
        for i in range(len(sourcePattern)):
            sourcePattern[i] = np.float(sourcePattern[i])
            
        sourceStatus = []
        for i in range(np.int(np.ceil( (maxTime+1)/np.size(sourcePattern)))):
            temp = np.copy(sourcePattern)
            temp[temp!=0.0] = np.random.normal(1.0,beamNoise)
            sourceStatus.extend(temp)
            
        bremsFluxTime = np.matrix(bremsFlux)*np.matrix(sourceStatus)

    sourceDetails = [bremsFlux,sourceStatus,sourceStrength]
        
    return bremsFluxTime, perfectFlux, sourceDetails
    
def chopper_run(matListFile,NRFDatabase,NRData,numBins=1000,energyMin=1e-3,energyMax=10,beamFlux=1e10,beamFreq=10.0,beamDutyFactor=1.0,beamNoise=0.01,totalTime=1.0,timeStep=0.001,chopperProfile='sqr',idealFiller=1.0):
    
    NRData = np.split(NRData,[1],axis=1)
    NREnergy = NRData[0] # Vector of energies that the database uses
    NRData = NRData[1]   # Non-resonant attenuation data by isotope, z = 1:100

    # ------ ------ ------
    # Extract Object and Foil descriptions
    # ------ ------ ------
    matList, nDensList, thickList, freqList = parse_materials(matListFile)
    matDetails = [matList, nDensList, thickList, freqList]
    if matList == []:
        sys.exit("\matlist ERROR: Foil and object description is empty or all commented out")
    nDensList = np.array(nDensList)
    thickList = np.array(thickList)
    freqList = np.array(freqList)
    
    # ------ ------ ------
    # Make list of NRF resonances and emissions
    # ------ ------ ------
    emitList = []
    counter = -1
    ZList = np.array(matList)[:,0]
    AList = np.array(matList)[:,1]
        
    # Check NRF database and pull out any lines which correspond to the materials in the problem and the energy window of interest
    for line in NRFDatabase:
        line = line.split(" ")
        [z, a] = map(int, [line[0], line[1]]);
        [Elevel, Egamma, Width, prob, GSprob, J0, Jr, TDebye] = map(float, [line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9]])

        if Width > 0 and ~np.isinf(Width) and prob > 0 and [z,a] in matList and Egamma>energyMin and Egamma<energyMax and Elevel>energyMin and Elevel<energyMax:
                nDens = nDensList[np.multiply(ZList==z,AList==a)][0]
                thickness = thickList[np.multiply(ZList==z,AList==a)][0]
                counter += 1
                x = NRFGamma(z,a,Elevel,Egamma,Width,prob,GSprob,J0,Jr,TDebye,nDens,thickness,counter)          
                emitList.append(x)    
    
    # ------ ------ ------
    # Discretize energy space
    # ------ ------ ------
    energyBins = np.logspace(np.log10(energyMin),np.log10(energyMax)-0.0000001,num=numBins)
    # Add NRF resonance energies for notches
    # Also add higher and lower energies so that it doesn't integrate too broadly
    # FIXME Using 10eV resonance for now but should calculate properly
    for line in emitList:
        energyBins = np.append(energyBins,line.Elevel-5e-6)
        energyBins = np.append(energyBins,line.Elevel)  
        energyBins = np.append(energyBins,line.Elevel+5e-6) 
    energyBins = np.unique(energyBins)                 # Order list and remove duplicates
    
    # ------ ------ ------
    # Discretize temporal space
    # ------ ------ ------
    maxTime = round(totalTime/timeStep)-1 # Number of time steps to be used, shifted to 0-index
    
    timeSeries = np.arange(0,totalTime,timeStep)
    
    # ------ ------ ------
    # Calculate microscopic cross sections for each material
    # ------ ------ ------
    matList_unique = [list(x) for x in set(tuple(x) for x in matList)] # To avoid repetitions, pull out unique [z,a] combos

    crossSection_micro_nonRes = np.zeros([np.size(energyBins),len(matList_unique)]) # matList_unique is used to index the cross sections
    crossSection_micro_NRF = np.copy(crossSection_micro_nonRes)
    
    counter = -1
    for mat in matList_unique:
        counter += 1
        crossSection_micro_nonRes[:,counter] = np.interp(energyBins,NREnergy[:,0],NRData[:,mat[0]-1])
        
        for line in emitList:
            if line.z == mat[0] and line.a==mat[1]:
                crossSection_micro_NRF[energyBins == line.Elevel,counter] += line.sigmaInt*line.prob
    
    # ------ ------ ------
    # Calculate macroscopic cross sections for each object
    # Objects being defined as materials with the same rotation frequency
    # ------ ------ ------
    freqList_unique = np.unique(freqList)
    
    attenuation_total = np.zeros([np.size(energyBins),maxTime+1])
       
    matList_np = np.array(matList)
    
    testObjectList = []
    for freq in freqList_unique[freqList_unique>=0.0]:
        y = testObject(freq, matList_np[freqList==freq], nDensList[freqList==freq], thickList[freqList==freq], idealFiller, matList_np[freqList==-freq], nDensList[freqList==-freq], thickList[freqList==-freq], matList_unique, crossSection_micro_nonRes, crossSection_micro_NRF, timeSeries, chopperProfile)
        
        testObjectList.append(y)

        attenuation_total = np.add(attenuation_total,y.crossSection_inTime_macro_total)
    
    # ------ ------ ------
    # Calculate calorimeter signal
    # ------ ------ ------
    bremsFluxTime, perfectFlux, sourceDetails = build_source(energyMax,energyBins,beamFlux,beamFreq,beamDutyFactor,beamNoise,totalTime,timeStep)
    
    flux = np.multiply( bremsFluxTime , np.exp(-attenuation_total) )
    fluxMean = np.multiply( perfectFlux , np.exp(-attenuation_total) )
    
    try: # Sample particular result from calculated averages
        fluxSample = np.random.poisson(flux)  
    except ValueError:
        fluxSample = np.round(np.random.normal(flux,np.sqrt(flux)+0.00001)) # Add small amount to standard deviation to ensure STD>0. Error removed by rounding
    
    calorimeter = np.trapz(np.multiply(energyBins[:,None],fluxSample), x=energyBins, axis=0).T   # Calculate average and particular calorimeter result
    calorimeterMean = np.trapz(np.multiply(energyBins[:,None],fluxMean), x=energyBins, axis=0).T # Use trapz due to uneven energy bin widths  
    
    calDiff = calorimeter-np.mean(calorimeter)  # Subtract the time-averaged calorimeter signal
    calDiffMean = calorimeterMean-np.mean(calorimeterMean)   
     
    # ------ Tidy up variables before returning ------
    crossSections = [crossSection_micro_nonRes, crossSection_micro_NRF]
    calorimeterOut = [calorimeter,  calorimeterMean]
    calDiffOut = [calDiff, calDiffMean]
    timeDetails = [maxTime, timeSeries, sourceDetails[1], bremsFluxTime]

    return calorimeterOut, calDiffOut, matDetails, energyBins, emitList, crossSections, sourceDetails, timeDetails, testObjectList
 
def chopper_plot(figNum,matList,energyBins,freqList,calorimeterOut,calDiffOut,xT,yT,mT,sourceDetails, timeDetails, testObjectList):
    
    # ------ Unpack variables from lists ------
    
    freqList_unique = np.unique( np.array(freqList) )
    
    bremsFlux = sourceDetails[0]

    maxTime = timeDetails[0]
    timeSeries = timeDetails[1]
    bremsFluxTime = timeDetails[3]
    
    calorimeter = calorimeterOut[0]
    calDiff = calDiffOut[0]
    
    # ------ Source and material data ------
    plt.figure(figNum)
    plt.clf()
    plt.subplot(221)
    plt.loglog(energyBins,bremsFlux,'k')
    plt.xlabel("Energy (MeV)")
    plt.ylabel(r"Flux ($\frac{1}{cm^2 s})$")
    plt.title("Bremsstrahlung source")
    
    legendString_moi = []
    legendString_filler = []
    for testObject in testObjectList:
        if testObject.freq == 0.0:
            plt.subplot(223)
            crossSectionTemp = np.sum( testObject.crossSection_moi_macro_NRF+testObject.crossSection_moi_macro_nonRes, axis=1 )
            if np.max(crossSectionTemp) == 0.0:
                plt.semilogx(energyBins,crossSectionTemp,'k')
            else:
                plt.loglog(energyBins,crossSectionTemp,'k') 
            
        else:
            plt.subplot(222)
            for i in range(len(testObject.matList)):
                crossSectionTemp = testObject.crossSection_moi_macro_NRF[:,i] + testObject.crossSection_moi_macro_nonRes[:,i]
                if np.max(crossSectionTemp) == 0.0:
                    plt.semilogx(energyBins,crossSectionTemp)
                else:
                    plt.loglog(energyBins,crossSectionTemp)
            legendString_moi.append( str(testObject.freq) )
                
            plt.subplot(224)
            for i in range(len(testObject.matList_filler)):
                crossSectionTemp = testObject.crossSection_filler_macro_NRF[:,i] + testObject.crossSection_filler_macro_nonRes[:,i]
                if np.max(crossSectionTemp) == 0.0:
                    plt.semilogx(energyBins,crossSectionTemp)
                else:
                    plt.loglog(energyBins,crossSectionTemp)
            legendString_filler.append( str(testObject.freq) )
    
    plt.subplot(223)
    plt.xlabel("Energy (MeV)")
    plt.ylabel(r"$\Sigma_{tot}$*Thickness ( - )")
    plt.title("Test object")
    
    plt.subplot(222)
    plt.xlabel("Energy (MeV)")
    plt.ylabel(r"$\Sigma_{tot}$*Thickness ( - )")
    plt.title("Chopper")
    plt.legend(legendString_moi)
    
    plt.subplot(224)
    plt.xlabel("Energy (MeV)")
    plt.ylabel(r"$\Sigma_{tot}$*Thickness ( - )")
    plt.title("Filler")
    plt.legend(legendString_filler)
    
    plt.tight_layout()
    
    # ------ Time varying signals ------ 
    plt.figure(figNum+1)
    plt.clf()
    
    plt.subplot(411)
    plt.plot(timeSeries,np.trapz(bremsFluxTime,x=energyBins, axis=0),'k')
    plt.xlabel('Time (s)')
    plt.ylabel(r"Flux ($\frac{1}{cm^{-2} s}$)")
    plt.title('Source flux')
    
    plt.subplot(412)
    for testObject in testObjectList:
        plt.plot(timeSeries,testObject.chopperStatus)
    plt.xlabel('Time (s)')
    plt.ylabel("Chopper atten.")
    plt.title('Chopper status')
    plt.ylim([-0.05,1.05])
    
    plt.subplot(413)
    plt.plot(timeSeries,calorimeter)
    plt.title("Total calorimeter signal")
    plt.ylabel(r"Deposited energy ($\frac{MeV}{cm^2}$)")
    plt.xlabel("Time (s)")
    
    plt.subplot(414)
    plt.plot(timeSeries,calDiff,'k')
    plt.title("Time-varying calorimeter signal")
    plt.ylabel(r"Deposited energy ($\frac{MeV}{cm^2}$)")
    plt.xlabel("Time (s)")
    
#    plt.tight_layout()
    
    # ------ Spectral analysis ------  
    plt.figure(figNum+2)
    plt.clf()
    plt.subplot(211)
    plt.plot(xT[0:int((maxTime+1)/2)],np.abs(yT[0:int((maxTime+1)/2)]),'k-')
    plt.xlabel("Frequency (Hz)")
    plt.title('Spectral analysis at all frequencies')
    
    freqList = np.array(freqList)
    
    plt.subplot(212)
    for freq in freqList_unique[freqList_unique>0.0]:
        plt.plot(freq*np.ones([100,1]),np.linspace(0,np.max(np.abs(yT[0:int((maxTime+1)/2)])),100),'--')
    plt.legend(legendString_moi)
    plt.plot(xT[0:int((maxTime+1)/2)],np.abs(yT[0:int((maxTime+1)/2)]),'k')
#    plt.plot(xT[0:int((maxTime+1)/2)],np.abs(mT[0:int((maxTime+1)/2)]),'r--')
    for freq in freqList_unique[freqList_unique>0.0]:
        x = np.where(xT[0:int((maxTime+1)/2)] == freq)
        plt.plot(freq,np.abs(yT[x]),'xr')
        plt.plot(freq,np.abs(mT[x]),'or')
    plt.xlim([0,np.max(freqList)*2.0])                 # Show frequencies up to twice the maximum frequency
    plt.xlabel("Frequency (Hz)")
    plt.title('Spectral analysis near expected frequencies')
    
    plt.tight_layout()
        
#    # ------ Flux plots for first cycle ------
#    timeIndex = np.floor(np.linspace(0,(maxTime+1)/np.min(freqList),9))
#    plt.figure(figNum+3)
#    plt.clf()
#    plt.suptitle("Flux measured at various time points")
#    for i in range(9):
#        plt.subplot(4,3,i+4)
#        plt.semilogy(energyBins,np.multiply(energyBins[:,None],flux[:,timeIndex[i]]))
#        plt.semilogy(energyBins,np.ones(np.shape(energyBins)),'r--')
#        plt.ylim(ymin=0.1)
#    timeIndex = timeIndex.astype(int)
#    plt.subplot(4,3,1)
#    plt.plot(timeSeries[timeSeries<(np.max(timeSeries[timeIndex])*2)],calDiff[timeSeries<(np.max(timeSeries[timeIndex])*2)],'k-')
#    plt.plot(timeSeries[timeIndex],calDiff[timeIndex],'ro')
#    
#    plt.subplot(4,3,2)
#    plt.plot(timeSeries,calDiff,'k-')
#    plt.plot(timeSeries[timeIndex],calDiff[timeIndex],'ro')
    
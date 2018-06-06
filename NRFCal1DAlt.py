# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 11:08:47 2016

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
        self.thickness = _thickness # [0]=warhead thickness, [1]=foil thickness
        self.index = _counter

        # Calculate the energy-integrated cross section
        g = (2.0*self.Jr+1)/(2.0*(2.0*self.J0+1))
        hbarc = 197.327e-15 # MeV m
        self.sigmaInt = 1.0e34 * 2.0 * (math.pi)**2 * g * (hbarc/self.Elevel)**2 * self.Width * self.prob * self.GSprob # eV b
        #print self.sigmaInt, self.Elevel, self.z,self.a
        self.sigmaInt = [self.sigmaInt/self.prob , self.sigmaInt/self.prob]
        
        # and the Doppler-broadened peak height
        Mc2 = self.a * 931.454 # MeV
        kB = 8.6173e-11 # MeV/K
        self.Delta = self.Elevel * math.sqrt(2*kB*300.0/Mc2) # FIXME this should use Teff
        self.sigmaDmax = 1.0e28 * 2.0 * (math.pi)**(3.0/2.0) * g * (hbarc/self.Elevel)**2 * self.prob * self.GSprob * self.Width / self.Delta # b

def parse_materials(fileName): 
    if type(fileName) != str:
        print('parse_material ERROR: \nMaterial list file name must be a string')
        return
        
    matList = []   # List of atomic and mass numbers for materials in object
    nDensList = [] # (atom / cm**3) * A * 1e-24 
    thickList = [] # [warhead,foil] thicknesses (cm)
    freqList = []  # Frequency of chopper wheel of that material
    
    matFile = open(fileName,'r')
    print(matFile)
    
    for isotope in matFile:
        isotope = isotope.strip()
        isoColumn = isotope.split(' ')
        
        if isoColumn[0]=="#": continue # Ignore comment lines in the material description
        
        matList.append([int(isoColumn[0]),int(isoColumn[1])])
        nDensList.append([float(isoColumn[2]),float(isoColumn[3])])
        thickList.append([float(isoColumn[4]),float(isoColumn[5])])
        freqList.append(float(isoColumn[6]))
        
    matFile.close()
    
    return matList, nDensList, thickList, freqList
    
def parse_materialsAlt(fileName): 
    if type(fileName) != str:
        print('parse_material ERROR: \nMaterial list file name must be a string')
        return
        
    matList = []   # List of atomic and mass numbers for materials in object
    nDensList = [] # (atom / cm**3) * A * 1e-24 
    thickList = [] # [warhead,foil] thicknesses (cm)
    freqList = []  # Frequency of chopper wheel of that material
    
    matFile = open(fileName,'r')
    print(matFile)
    
    for isotope in matFile:
        isotope = isotope.strip()
        isoColumn = isotope.split(' ')
        
        if isoColumn[0]=="#": continue # Ignore comment lines in the material description
        
        matList.append([int(isoColumn[0]),int(isoColumn[1])])
        nDensList.append([float(isoColumn[2]),float(isoColumn[3])])
        thickList.append([float(isoColumn[4]),float(isoColumn[5])])
        freqList.append(float(isoColumn[6]))
        
    matFile.close()
    
    return matList, nDensList, thickList, freqList
    
def build_source(energyMax,energyBins,beamFlux=1e10,beamFreq=10.0,beamDutyFactor=1.0,beamNoise=0.01,totalTime=1.0,timeStep=0.001):
    maxTime = int(totalTime/timeStep)-1 # Number of time steps to be used, shifted to 0-index

    # Build Bremsstrahlung source using Kramers law approximation
    # Kramers law: Intensity(E) = K * Z * (Emax - E)
    bremsFlux = np.subtract(energyMax,energyBins)      
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
    
    if beamNoise == 0.0: # No noise added to the source signal
        bremsFluxTime = np.copy(perfectFlux)
    else:
        for i in range(len(sourcePattern)):
            sourcePattern[i] = np.float(sourcePattern[i])
            
        sourceStatus = []
        for i in range(np.int(np.ceil( (maxTime+1)/np.size(sourcePattern)))):
            temp = np.copy(sourcePattern)
#            temp[temp!=0.0] = np.random.normal(1.0,beamNoise,len(temp[temp!=0.0]))
            temp[temp!=0.0] = np.random.normal(1.0,beamNoise)
            sourceStatus.extend(temp)
            
        bremsFluxTime = np.matrix(bremsFlux)*np.matrix(sourceStatus)
        
#            
#            
#            
#        sourceStatus = np.tile(sourcePattern, ) # Repeat source unit/cycle pattern. This may produce extra time steps
#        sourceStatus = sourceStatus[0:maxTime+1]                                            # Truncate vector to desired total time
#     
#        perfectFlux = np.matrix(bremsFlux)*np.matrix(sourceStatus)
#    
#        try:
#            sourceStatus[sourceStatus!=0] = np.random.normal(1.0, beamNoise, len(sourceStatus[sourceStatus!=0]))
#            bremsFluxTime = np.matrix(bremsFlux)*np.matrix(sourceStatus)
#        except ValueError:
#            print(np.shape(bremsFluxTime))
#            print(np.shape(perfectFlux))
            
        # bremsFluxTime = np.round(np.random.normal( perfectFlux,perfectFlux*beamNoise )) # Sample the source strength as a normal random variable
        # bremsFluxTime = np.round(np.random.normal( bremsFluxTime,np.sqrt(bremsFluxTime) ))
    
    sourceDetails = [bremsFlux,sourceStatus,sourceStrength]
        
    return bremsFluxTime, perfectFlux, sourceDetails
    
def chopper_run(matListFile,NRFDatabase,NRData,numBins=1000,energyMin=1e-3,energyMax=10,beamFlux=1e10,beamFreq=10.0,beamDutyFactor=1.0,beamNoise=0.01,totalTime=1.0,timeStep=0.001,foilProfile='sqr'):
    
    NRData = np.split(NRData,[1],axis=1)
    NREnergy = NRData[0] # Vector of energies that the database uses
    NRData = NRData[1]   # Non-resonant attenuation data by isotope, z = 1:100

    # ------ ------ ------
    # Extract Object and Foil descriptions
    # ------ ------ ------
    matList, nDensList, thickList, freqList = parse_materials(matListFile)
    matDetails = [matList, nDensList, thickList, freqList]
    if matList == []:
        sys.exit("\Foil and object description is empty or all commented out")
    numMat = len(matList)
    nDensList = np.array(nDensList)
    thickList = np.array(thickList)
    
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
    # Calculate total cross section for the foils and object
    # ------ ------ ------
    crossSectionFoil = np.zeros([np.size(energyBins),int(np.size(matList)/2)])
    crossSectionObject = np.zeros([np.size(energyBins),1])
    crossSectionFiller = np.zeros([np.size(energyBins),int(np.size(matList)/2)])   
          
    counter = -1
    for mat in matList:
        if numMat == 1:
            nDensObject = nDensList[0][0]
            nDensFoil = nDensList[0][1]
        else:
            nDensObject = nDensList[np.multiply(ZList==mat[0],AList==mat[1])][0][0]
            nDensFoil = nDensList[np.multiply(ZList==mat[0],AList==mat[1])][0][1]
            
        # Build non-resonant cross sections from X-COM database
        # Linearly interpolate the data for the specified energy bins
        counter += 1
        crossSectionFoil[:,counter] = np.multiply(nDensFoil*thickList[counter][1],np.interp(energyBins,NREnergy[:,0],NRData[:,mat[0]-1]))
        crossSectionObject = np.add(crossSectionObject,np.multiply(nDensObject*thickList[counter][0],np.interp(energyBins,NREnergy[:,0],NRData[:,mat[0]-1]))[:,None])
        crossSectionFiller[:,counter] = np.multiply(nDensFoil*thickList[counter][1],np.interp(energyBins,NREnergy[:,0],NRData[:,mat[0]-1]))
        
        # Add in the NRF lines
        # Note the filler material is skipped in this idealized case
        for line in emitList:
            if line.z == mat[0] and line.a==mat[1]:
                crossSectionFoil[energyBins == line.Elevel,counter] += np.multiply(line.sigmaInt[1],nDensFoil*line.prob*thickList[counter][1])
                crossSectionObject[energyBins == line.Elevel] += np.multiply(line.sigmaInt[0],nDensObject*line.prob*thickList[counter][0])
    
    # ------ ------ ------
    # Calculate time dependent results
    # ------ ------ ------
    maxTime = int(totalTime/timeStep)-1 # Number of time steps to be used, shifted to 0-index
    
    timeSeries = np.arange(0,totalTime,timeStep)
    foilStatus = np.ones([maxTime+1,numMat])
    
    foilArgument = np.zeros([np.size(energyBins),maxTime+1])
    
    # Calculate status vector of when each material is in the beam and add cross section to the 
    # Can currently have square or sine-wave chopper profile
    if foilProfile == 'sqr':
        for i in range(numMat):
            foilStatus[:,i] = np.round( (1.0 + np.sin(freqList[i]*2.0*np.pi * timeSeries) )/2 )
    
            foilArgument = np.add(foilArgument,np.matrix(crossSectionFoil[:,i]).T*np.matrix(foilStatus[:,i]))
            foilArgument = np.add(foilArgument,np.matrix(crossSectionFiller[:,i]).T*np.matrix(1.0-foilStatus[:,i]))
    elif foilProfile == 'sin':
        for i in range(numMat):
            foilStatus[:,i] = ( (1.0 + np.sin(freqList[i]*2.0*np.pi * timeSeries) )/2 )
    
            foilArgument = np.add(foilArgument,np.matrix(crossSectionFoil[:,i]).T*np.matrix(foilStatus[:,i]))
            foilArgument = np.add(foilArgument,np.matrix(crossSectionFiller[:,i]).T*np.matrix(1.0-foilStatus[:,i]))
        
    # ------ ------ ------
    # Calculate calorimeter signal
    # ------ ------ ------
    bremsFluxTime, perfectFlux, sourceDetails = build_source(energyMax,energyBins,beamFlux,beamFreq,beamDutyFactor,beamNoise,totalTime,timeStep)
    
    flux = np.multiply(bremsFluxTime,np.exp(-crossSectionObject-foilArgument))
    fluxMean = np.multiply(perfectFlux,np.exp(-crossSectionObject-foilArgument))
    
    try: # Sample particular result from calculated averages
        fluxSample = np.random.poisson(flux)  
    except ValueError:
        fluxSample = np.round(np.random.normal(flux,np.sqrt(flux)+0.00001)) # Add small amount to standard deviation to ensure STD>0. Error removed by rounding
    
    calorimeter = np.trapz(np.multiply(energyBins[:,None],fluxSample), x=energyBins, axis=0).T   # Calculate average and particular calorimeter result
    calorimeterMean = np.trapz(np.multiply(energyBins[:,None],fluxMean), x=energyBins, axis=0).T # Use trapz due to uneven energy bin widths  
    
    #calDiff = calorimeter - calorimeter[0]           # Subtract the 'background' from the calorimeter signal
    #calDiffMean = calorimeterMean-calorimeterMean[0] # Can subtract first value or mean value
    
    calDiff = calorimeter-np.mean(calorimeter)  
    calDiffMean = calorimeterMean-np.mean(calorimeterMean)   
     
    # ------ Tidy up variables before returning ------
    crossSections = [crossSectionObject, crossSectionFoil, crossSectionFiller]
    calorimeterOut = [calorimeter,  calorimeterMean]
    calDiffOut = [calDiff, calDiffMean]
    timeDetails = [maxTime, timeSeries, foilStatus, sourceDetails[1], bremsFluxTime]
    
    return calorimeterOut, calDiffOut, matDetails, energyBins, emitList, crossSections, sourceDetails, timeDetails
 
def chopper_plot(figNum,matList,energyBins,freqList,calorimeterOut,calDiffOut,xT,yT,mT,crossSections,sourceDetails, timeDetails):
    
    
    # ------ Unpack variables from lists ------
    crossSectionObject = crossSections[0]
    crossSectionFoil = crossSections[1]
    crossSectionFiller = crossSections[2]    
    
    bremsFlux = sourceDetails[0]

    maxTime = timeDetails[0]
    timeSeries = timeDetails[1]
    foilStatus = timeDetails[2]
    bremsFluxTime = timeDetails[4]
    
    calorimeter = calorimeterOut[0]
    calDiff = calDiffOut[0]
    
    numMat = len(matList)

    # ------ Source and material data ------
    plt.figure(figNum)
    plt.clf()
    plt.subplot(221)
    plt.loglog(energyBins,bremsFlux,'k')
    plt.xlabel("Energy (MeV)")
    plt.ylabel(r"Flux ($\frac{1}{cm^2 s})$")
    plt.title("Bremsstrahlung source")
    
    plt.subplot(223)
    if np.max(crossSectionObject) == 0.0:
        plt.semilogx(energyBins,crossSectionObject,'k')
    else:
        plt.loglog(energyBins,crossSectionObject,'k')    
    plt.xlabel("Energy (MeV)")
    plt.ylabel(r"$\Sigma_{tot}$*Thickness ( - )")
    plt.title("Test object")
    
    plt.subplot(222)
    counter = -1
    if np.max(crossSectionFoil) == 0.0:
        for mat in matList:
            counter += 1   
            plt.semilogx(energyBins,crossSectionFoil[:,counter])
    else:
        for mat in matList:
            counter += 1   
            plt.loglog(energyBins,crossSectionFoil[:,counter])
    plt.xlabel("Energy (MeV)")
    plt.ylabel(r"$\Sigma_{tot}$*Thickness ( - )")
    plt.title("Chopper")
    legendString = str(matList)
    legendString = legendString.split("], [")
    legendString[0] = legendString[0][2:len(legendString[0])]
    legendString[len(legendString)-1] = legendString[len(legendString)-1][0:len(legendString[len(legendString)-1])-2]
    plt.legend(legendString)
    
    plt.subplot(224)
    counter = -1
    if np.max(crossSectionFiller) == 0.0:
        for mat in matList:
            counter += 1   
            plt.semilogx(energyBins,crossSectionFiller[:,counter])
    else:
        for mat in matList:
            counter += 1   
            plt.loglog(energyBins,crossSectionFiller[:,counter])
    plt.xlabel("Energy (MeV)")
    plt.ylabel(r"$\Sigma_{tot}$*Thickness ( - )")
    plt.title("Filler")
    legendString = str(matList)
    legendString = legendString.split("], [")
    legendString[0] = legendString[0][2:len(legendString[0])]
    legendString[len(legendString)-1] = legendString[len(legendString)-1][0:len(legendString[len(legendString)-1])-2]
    plt.legend(legendString)
    
    plt.tight_layout()
    
    # ------ Time varying signals ------ 
    plt.figure(figNum+1)
    plt.clf()
#    plt.suptitle("Time varying signals")
    
    plt.subplot(411)
    plt.plot(timeSeries,np.trapz(bremsFluxTime,x=energyBins, axis=0),'k')
    plt.xlabel('Time (s)')
    plt.ylabel(r"Flux ($\frac{1}{cm^{-2} s}$)")
    plt.title('Source flux')
    
    plt.subplot(412)
    for i in range(numMat):
        plt.plot(timeSeries,foilStatus[:,i])
    plt.xlabel('Time (s)')
    plt.ylabel("Chopper atten.")
    plt.title('Chopper status')
    plt.legend(legendString)
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
    
    plt.subplot(212)
    for freq in freqList:
        plt.plot(freq*np.ones([100,1]),np.linspace(0,np.max(np.abs(yT[0:int((maxTime+1)/2)])),100),'--')
    plt.legend(legendString)
    plt.plot(xT[0:int((maxTime+1)/2)],np.abs(yT[0:int((maxTime+1)/2)]),'k')
#    plt.plot(xT[0:int((maxTime+1)/2)],np.abs(mT[0:int((maxTime+1)/2)]),'r--')
    for freq in freqList:
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
    
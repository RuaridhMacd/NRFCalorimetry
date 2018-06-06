# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 16:27:21 2016

@author: Ruaridh
"""
import numpy as np

def fourierCheck(_testObject, _energyBins, _testObjectList, _sourceDetails):
    # This function calculates the result of the Fourier transform at the frequencies of interest using the actual data
    # We can then use this result to estimate the maximum possible achievable accuracy

    testObject = _testObject
    testObjectList = _testObjectList
    energyBins = _energyBins
    bremsFlux = _sourceDetails[0]
    beamDutyFactor = _sourceDetails[3]
    
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
            
    energyTerm = energyList*bremsFlux[energyIndex][:,0]*beamDutyFactor*1e-5
    foilNRTerm = np.exp( -np.sum( testObject.crossSection_moi_macro_nonRes[energyIndex], axis=1 ) )
    foilNRFTerm = ( 1.0-np.exp(-np.sum( testObject.crossSection_moi_macro_NRF[energyIndex], axis=1 )) )

#    print(np.shape(energyTerm))
#    print(np.shape(foilNRTerm))
#    print(np.shape(foilNRFTerm))
    
    fourierCoeff = itemTerm*objectTerm*energyTerm*foilNRTerm*foilNRFTerm*2.0/np.pi # Store coefficients for calculations - [other,Object]

    return np.array(fourierCoeff), energyList
    
def calcMCoeffs(_testObject, _energyBins, _testObjectList, _matList_unique, _crossSections, _sourceDetails):
    
    testObject = _testObject
    testObjectList = _testObjectList
    energyBins = _energyBins
    
    bremsFlux = _sourceDetails[0]
    beamDutyFactor = _sourceDetails[3]

    crossSection_micro_nonRes = _crossSections[0]
    crossSection_micro_NRF = _crossSections[1]
    
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
            
    energyTerm = energyList*bremsFlux[energyIndex][:,0]*beamDutyFactor*1e-5
    foilNRTerm = np.exp( -np.sum( testObject.crossSection_moi_macro_nonRes[energyIndex], axis=1 ) )
    foilNRFTerm = ( 1.0-np.exp(-np.sum( testObject.crossSection_moi_macro_NRF[energyIndex], axis=1 )) )
    
    frontCoeff = itemTerm*energyTerm*foilNRTerm*foilNRFTerm*2.0/np.pi
             
    return frontCoeff,varCoeff
    
def func2Solve(linDensGuess,yTPeakList,frontCoeffList,varCoeffList):
    
    solution = np.ones(np.shape(linDensGuess)) 
    yTPeakList = np.array(yTPeakList)
    
    for i in range(len(linDensGuess)):
        
        solution[i] = np.abs( yTPeakList[i] - np.sum( np.multiply( frontCoeffList[i],np.exp( -np.sum( np.multiply(np.array(linDensGuess) , varCoeffList[i]) ,1) ) ) ,0) )
             
    return solution

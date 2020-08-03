#!/bin/env python
# This script is intended for fully-implicit simulation of 1-dimensional oil-water system #

import math
import time
import numpy as np
from matplotlib import pyplot as plt

############################### FUNCTIONS DEFINITION ######################################

def relPerm(Sw):
    "Function to calculate water [0] and oil [1] relative permeability based on Sw of a gridblock."
    normSw = (Sw-Swirr)/(1.0-Sorw-Swirr)
    krWater = krwo*normSw**coreyNw
    krOil = (1-normSw)**coreyNo
    return krWater,krOil

def derivRelPerm(Sw):
    "Function to calculate gradient of water [0] and oil [1] relative permeability based on Sw of a gridblock."
    normSw = (Sw-Swirr)/(1.0-Sorw-Swirr)
    dkrWater = (krwo/(1-Sorw-Swirr))*coreyNw*(normSw**(coreyNw-1))
    dkrOil = (coreyNo/(1-Sorw-Swirr))*((1-normSw)**(coreyNw-1))
    return dkrWater,dkrOil

def determineConnection(NGrid):
    "Function to determine connecting gridblocks of a gridblock."
    mConnectionL = np.zeros((NGrid),dtype=int)
    mConnectionR = np.zeros((NGrid),dtype=int)
    for n in range(NGrid):
        mConnectionL[n] = n-1
        mConnectionR[n] = n+1
    mConnectionL[0] = -1
    mConnectionR[NGrid-1] = -1
    mConnections = np.array([mConnectionL,mConnectionR])
    return mConnections

def darcyFlow(Perm,Visc,Pin,Pout):
    "Function to calculate flow rate according to Darcy's Equation."
    return (Perm/Visc)*AGrid*(Pin-Pout)

def calculateTransmissibility(Sw,iGrid):
    "Function to calculate water [0] and oil [1] transmissibilities to all connecting gridblocks."
    fGeo = cPerm * (AGrid/xGrid)
    fViscOil = 1/(cMuo*refBo)
    fViscWat = 1/(cMuw*refBw)
    fRelPermWat = np.zeros((2))
    fRelPermOil = np.zeros((2))
    if iGrid==0:
        fRelPermWat[1] = relPerm(Sw[iGrid])[0]
        fRelPermOil[1] = relPerm(Sw[iGrid])[1]
    elif iGrid==NGrid-1:
        fRelPermWat[0] = relPerm(Sw[iGrid-1])[0]
        fRelPermOil[0] = relPerm(Sw[iGrid-1])[1]
    else:
        fRelPermWat[0] = relPerm(Sw[iGrid-1])[0]
        fRelPermWat[1] = relPerm(Sw[iGrid])[0]
        fRelPermOil[0] = relPerm(Sw[iGrid-1])[1]
        fRelPermOil[1] = relPerm(Sw[iGrid])[1]
    TransOil = np.array([fGeo*fViscOil*fRelPermOil[0],fGeo*fViscOil*fRelPermOil[1]])
    TransWater = np.array([fGeo*fViscWat*fRelPermWat[0],fGeo*fViscWat*fRelPermWat[1]])
    return TransWater,TransOil

def fillFlowVectorElement(Sw,Po,phase,iGrid):
    "Function to fill an element of the Flow Vector."
    fVectorElement = 0.
    for m in range(len(mConnections)):
        if mConnections[m][iGrid] < 0:
            fVectorElement += 0.
        else:
            dPressure = Po[mConnections[m][iGrid]]-Po[iGrid]
            Trans = calculateTransmissibility(Sw,iGrid)
            fVectorElement += Trans[phase][m] * dPressure
    return fVectorElement

def fillFlowMatrixDiagElement(Sw,Po,phase,iGrid):
    "Function to fill a diagonal element of the Flow Matrix."
    fMatrixElement = 0.
    for m in range(len(mConnections)):
        im = mConnections[m][iGrid]
        if im < 0 or im > iGrid:
            fMatrixElement += 0.
        else:
            Swm = Sw[im]
            Pom = Po[im]
            dPressure = Pom-Po[iGrid]
            if relPerm(Swm)[phase] == 0:
                dTrans = 0.
            else:
                dTrans = calculateTransmissibility(Sw,iGrid)[phase][m]*derivRelPerm(Swm)[phase]/relPerm(Swm)[phase]
            fMatrixElement += dTrans * dPressure
    return fMatrixElement

def fillFlowMatrixNondiagElement(Sw,Po,phase,iGrid,mGrid):
    "Function to fill a non-diagonal element of the Flow Matrix."
    dPressure = Po[mGrid] - Po[iGrid]
    Swm = Sw[mGrid]
    if mGrid > iGrid or relPerm(Swm)[phase] == 0:
        fMatrixElement = 0.
    else:
        mIdx = (mGrid+1-iGrid)/2
        dTrans = calculateTransmissibility(Sw,iGrid)[phase][mIdx]*derivRelPerm(Swm)[phase]/relPerm(Swm)[phase]
        fMatrixElement = dTrans * dPressure
    return fMatrixElement

def fillUnknownVector(Sw,Po):
    "Function to build the Unknown Vector consisting of Sw and Po."
    xVector = np.zeros((NGrid,1,2))
    xVector[:,0,0] = Sw
    xVector[:,0,1] = Po
    return xVector

def fillResidualVectors(Sw,Po):
    "Function to build the Residual Vectors consisting of Flow Vector [0], Accumulation Vector [1], and Sink/Source Vector [2]."
    qVector = np.zeros((NGrid,1,2))
    cVector = np.zeros((NGrid,1,2))
    fVector = np.zeros((NGrid,1,2))
    for n in range(NGrid):
        fVector[n][0][0] = fillFlowVectorElement(Sw,Po,0,n)
        fVector[n][0][1] = fillFlowVectorElement(Sw,Po,1,n)
        if n==0 or n==NGrid-1: # for well gridblocks, assume transmissibility 1000 times that of reservoir gridblocks
            fVector[n,:,:] *= 1e3
    cVector[:,0,0] = (AGrid*xGrid/dTime)*(cPoro*Sw/refBw)
    cVector[:,0,1] = (AGrid*xGrid/dTime)*(cPoro*(1-Sw)/refBo)
    cVector[0,:,:] *= (0.5/cPoro) # for well gridblocks, assume porosity of 0.5
    cVector[-1,:,:] *= (0.5/cPoro)
    qVector[0][0][0] = darcyFlow(cPerm*1e3*relPerm(Sw[0])[0],cMuw,BHPInj,Po[0])
    # for well gridblocks, assume permeability 1000 times that of reservoir gridblocks
    qVector[NGrid-1][0][0] = -1.0*darcyFlow(cPerm*1e3*relPerm(Sw[NGrid-1])[0],cMuw,Po[NGrid-1],BHPProd) 
    qVector[NGrid-1][0][1] = -1.0*darcyFlow(cPerm*1e3*relPerm(Sw[NGrid-1])[1],cMuo,Po[NGrid-1],BHPProd)
    return fVector,cVector,qVector

def fillJacobianMatrix(Sw,Po):
    "Function to build the Jacobian Matrix consisting of Flow Matrix [0], Accumulation Matrix [1], and Sink/Source Matrix [2]."
    qMatrix = np.zeros((NGrid,NGrid,2,2))
    cMatrix = np.zeros((NGrid,NGrid,2,2))
    fMatrix = np.zeros((NGrid,NGrid,2,2))
    for n in range(NGrid):
        cMatrix[n][n][0][0] = (AGrid*xGrid/dTime)*(cPoro/refBw)
        cMatrix[n][n][0][1] = (AGrid*xGrid/dTime)*Sw[n]*(cPoro*compw/refBw)
        cMatrix[n][n][1][0] = (-AGrid*xGrid/dTime)*(cPoro/refBo)
        cMatrix[n][n][1][1] = (AGrid*xGrid/dTime)*(1-Sw[n])*(cPoro*compo/refBo)
        cMatrix[0,0,:,0] *= (0.5/cPoro) # for well gridblocks, assume porosity of 0.5
        cMatrix[-1,-1,:,0] *= (0.5/cPoro)
        if n==NGrid-1:
            if Sw[n] >= Swc and Sw[n] <= 1-Sorw: # for well gridblocks, assume permeability 1000 times that of reservoir gridblocks
                qMatrix[0][0][0][0] = (cPerm*1e3*AGrid/cMuw)*(BHPInj-Po[0])*(derivRelPerm(Sw[0])[0])
                qMatrix[0][0][0][1] = -(cPerm*1e3*relPerm(Sw[0])[0]*AGrid/cMuw)
                qMatrix[n][n][0][0] = -(cPerm*1e3*AGrid/cMuw)*(Po[n]-BHPProd)*(derivRelPerm(Sw[n])[0])
                qMatrix[n][n][1][0] = (AGrid*cPerm*1e3/cMuo)*(Po[n]-BHPProd)*(derivRelPerm(Sw[n])[1])
                qMatrix[n][n][0][1] = -(cPerm*1e3*relPerm(Sw[n])[0]*AGrid/cMuw)
                qMatrix[n][n][1][1] = -(cPerm*1e3*relPerm(Sw[n])[1]*AGrid/cMuo)
    for n in range(NGrid): 
        for m in range(NGrid):
            if n==m:
                fMatrix[n][m][0][0] = fillFlowMatrixDiagElement(Sw,Po,0,n)
                fMatrix[n][m][0][1] = (-1)*np.sum(calculateTransmissibility(Sw,n)[0])
                fMatrix[n][m][1][0] = fillFlowMatrixDiagElement(Sw,Po,1,n)
                fMatrix[n][m][1][1] = (-1)*np.sum(calculateTransmissibility(Sw,n)[1])
                if n==0 or n==NGrid-1: # for well gridblocks, assume transmissibility 1000 times that of reservoir gridblocks
                    fMatrix[n,m,:,:] *= 1e3
            else:
                if m in mConnections[:,n]:
                    mIdx = (m+1-n)/2
                    fMatrix[n][m][0][0] = fillFlowMatrixNondiagElement(Sw,Po,0,n,m)
                    fMatrix[n][m][0][1] = calculateTransmissibility(Sw,n)[0][mIdx]
                    fMatrix[n][m][1][0] = fillFlowMatrixNondiagElement(Sw,Po,1,n,m)
                    fMatrix[n][m][1][1] = calculateTransmissibility(Sw,n)[1][mIdx]
                    if n==0 or n==NGrid-1: # for well gridblocks, assume transmissibility 1000 times that of reservoir gridblocks
                        fMatrix[n,m,:,:] *= 1e3
    return fMatrix,cMatrix,qMatrix

def NextIterDeltaUnknown(RV,JM,cVC):
    "Function to calculate delta unknown vector for one Newton linear iteration."
    blockJacobianMatrix = JM[0] - JM[1] + JM[2]
    blockResidualVector = (RV[1]-cVC) - RV[0] - RV[2]
    jacobianMatrix = np.zeros((NGrid*2,NGrid*2))
    residualVector = np.zeros((NGrid*2))
    for i in range(NGrid):
        residualVector[i*2] = blockResidualVector[i][0][0]
        residualVector[i*2+1] = blockResidualVector[i][0][1]
        for j in range(NGrid):
            jacobianMatrix[i*2][j*2] = blockJacobianMatrix[i][j][0][0]
            jacobianMatrix[i*2][j*2+1] = blockJacobianMatrix[i][j][0][1]
            jacobianMatrix[i*2+1][j*2] = blockJacobianMatrix[i][j][1][0]
            jacobianMatrix[i*2+1][j*2+1] = blockJacobianMatrix[i][j][1][1]
    jacobiInv = np.linalg.inv(jacobianMatrix)
    deltaUnknownNew = np.matmul(jacobiInv,residualVector)
    blockDeltaUnknown = np.zeros((NGrid,1,2))
    for i in range(NGrid):
        blockDeltaUnknown[i][0][0] = deltaUnknownNew[i*2]
        blockDeltaUnknown[i][0][1] = deltaUnknownNew[i*2+1]
    return blockDeltaUnknown

def NextTimestepUnknown(Sw,Po):
    "Function to calculate the objective unknowns (Sw and Po) for the next timestep."
    SwInit = Sw
    PoInit = Po
    cVectorCurrent = fillResidualVectors(SwInit,PoInit)[1]
    currentUnknown = fillUnknownVector(SwInit,PoInit)
    Err = 1.0
    idx = 0
    while Err >= 1e-4 and idx < 20:
        Err = 0
        RVElements = fillResidualVectors(Sw,Po)
        JMElements = fillJacobianMatrix(Sw,Po)
        deltaUnknown = NextIterDeltaUnknown(RVElements,JMElements,cVectorCurrent)
        newUnknown = currentUnknown + deltaUnknown
        Sw = newUnknown[:,0,0]
        Po = newUnknown[:,0,1]
        for n in range(NGrid):
            if newUnknown[n][0][0] < Swc:
                newUnknown[n][0][0] = Swc
            if newUnknown[n][0][0] > 1-Sorw:
                newUnknown[n][0][0] = 1-Sorw
            Err += math.sqrt((currentUnknown[n][0][0]-newUnknown[n][0][0])**2)
        currentUnknown = newUnknown
        idx += 1
    print 'No. of linear iteration:',idx
    return newUnknown

################################### MAIN CODE ##########################################

# Rel. Perm. Curve
Swirr = 0.20
Sorw = 0.20
krwo = 0.40
coreyNo = 2.0
coreyNw = 2.0

# Reservoir Initialization
NGrid = 102
xGrid = 2.0 # m (1 m = 3.281 ft)
AGrid = xGrid**2 # cube grid blocks
cPerm = 1e-13 # m2 (1e-15 m2 = 1 mD)
cMuo = 3e-3 # Pa.s (1e-3 Pa.s = 1.0 cp)
cMuw = 1e-3 # Pa.s (1e-3 Pa.s = 1.0 cp)
cPoro = 0.20
Swc = 0.20
Poi = 1.01e7 # Pa (1e5 Pa = 14.50 psi)
refBw = 1.01 # valid for pressure 50-250 bar
refBo = 1.02 # valid for pressure 55-550 bar
compw = 1e-9 # 1/Pa (1e-5/Pa = 1/bar)
compo = 2e-10 # 1/Pa (1e-5/Pa = 1/bar)

Sw = np.full((NGrid),Swc)
Po = np.full((NGrid),Poi)
Sw[0] = 1-Sorw

mConnections = determineConnection(NGrid)

# Well Initialization
BHPInj = 2e7 # Pa (1e5 Pa = 14.50 psi)
BHPProd = 1e7 # Pa (1e5 Pa = 14.50 psi)

# Simulation Specifications
TimeStep = 0.5 # days
dTime = TimeStep*86400 # convert to seconds

#################################### TESTING AREA #########################################

# Print and Plot Sw and Po over time
NSteps = 200 # Number of timesteps for the simulation
SaturationWater = np.zeros((NSteps+1,NGrid))
PressureGrid = np.zeros((NSteps+1,NGrid))
idxGrid = np.arange(NGrid)

TStep0 = fillUnknownVector(Sw,Po)
SaturationWater[0] = Sw
PressureGrid[0] = Po*1e-5

totaltic = time.time()
for i in range(NSteps):
    print 'Timestep',i+1
    tic = time.time()
    newUnknown = NextTimestepUnknown(Sw,Po)
    toc = time.time()
    print 'Time elapsed:',toc-tic,'s \n'
    Sw[:] = newUnknown[:,0,0]
    Po[:] = newUnknown[:,0,1]
    SaturationWater[i+1] = Sw
    PressureGrid[i+1] = Po*1e-5
totaltoc = time.time()
print 'Total time of simulation',totaltoc-totaltic,'s \n'

plt.figure()
cmap = plt.get_cmap('gnuplot')
plt.title('Saturation Distribution Over Time')
for i in range(NSteps):
    if i==0 or (i+1)%(NSteps/10) == 0:
        plt.plot(idxGrid,SaturationWater[i],color=cmap(100*i/NSteps)) 
plt.xlabel('Grid Index') ; plt.ylabel('s_w, $fraction$')
plt.ylim([0,1])
plt.legend(loc='lower left')

plt.figure()
plt.title('Pressure Distribution Over Time')
for i in range(NSteps):
    if i==0 or (i+1)%(NSteps/10) == 0:
        plt.plot(idxGrid,PressureGrid[i],color=cmap(100*i/NSteps)) 
plt.xlabel('Grid Index') ; plt.ylabel('Pressure, $bar$')
plt.legend(loc='lower left')
plt.show()

# Plot kr curve
SwVals = np.linspace(Swirr,1-Sorw,num=50)
SwPlot = SwVals
kroPlot = relPerm(SwVals)[1]
krwPlot = relPerm(SwVals)[0]

plt.figure()
plt.title('Oil-Water Relative Permeability Plot')
plt.plot(SwPlot,kroPlot,color='r',label=r'kro')
plt.plot(SwPlot,krwPlot,color='b',label=r'krw')
plt.xlabel('Saturation, $s_w$') ; plt.ylabel('Rel. perm., $fraction$')
plt.xlim([0,1]); plt.ylim([0,1])
plt.legend(loc='lower right')
# plt.show()
import numpy as np
import scipy as sp
import pandas as pd
import math

from sklearn.metrics import mean_squared_error, r2_score

################################################################################
################################################################################

def calc_performance(predY, obsY):
    if isinstance(predY, pd.DataFrame):
        thisPred = predY.values
        thisObs = obsY.values
    else:
        thisPred = predY
        thisObs = obsY

    nanhold = np.isnan(thisObs)
    if (thisObs[~nanhold].size == 0) or (thisPred[~nanhold].size == 0) or any(np.isnan(thisPred)):
        outRMSE, outNMSE, outR, outR2 = np.nan, np.nan, np.nan, np.nan
    else:
        outRMSE = math.sqrt(mean_squared_error(thisObs[~nanhold], thisPred[~nanhold]))
        outR2 = r2_score(obsY[~nanhold], predY[~nanhold])
        # outR = math.sqrt(np.abs(outR2))*np.sign(outR2)
        _, _, r_value, _, _ = sp.stats.linregress(thisObs[~nanhold], thisPred[~nanhold])
        outR = r_value
        #have to demean the data for true variance calc
        outNMSE = np.sum(np.power(thisObs[~nanhold] - thisPred[~nanhold],2))/np.sum(np.power(thisObs[~nanhold]-np.mean(thisObs[~nanhold]),2))

    outStats = {
        'RMSE': outRMSE,
        'NMSE': outNMSE,
        'r': outR,
        'R2': outR2,
    }
    return outStats

################################################################################
################################################################################

def calc_CV_performance(predY):
    modNames = [_ for _ in predY.columns if not _ in ['Observed','Case']]
    testSet = predY[predY['Case']=='val']
    # setup the dataFrame
    outStats = pd.DataFrame(columns=['RMSE','NMSE','r','Name'])
    for ii, thisMod in enumerate(modNames):
        thisRes = calc_performance(testSet[thisMod].values,testSet['Observed'].values)
        outStats.loc[ii,'RMSE'] = thisRes['RMSE']
        outStats.loc[ii,'NMSE'] = thisRes['NMSE']
        outStats.loc[ii,'r'] = thisRes['r']
        outStats.loc[ii,'Name'] = thisMod
    return outStats

################################################################################
################################################################################
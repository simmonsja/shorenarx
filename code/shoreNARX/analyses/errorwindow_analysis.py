import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

################################################################################
################################################################################

def run_errorwindow_analysis(trainObjCollection: dict):
    #create the holding tank
    dataOut = {}
    for ii, key in enumerate(trainObjCollection.keys()):
        print('Running {}:'.format(key))
        thisTObj = trainObjCollection[key]
        # Load the data details
        runsNum = thisTObj.statsOut.__len__()
        # runsNum = 1
        numFolds = thisTObj.statsOut[0].__len__()

        # easy way to create a correct size dummy is to wait until now
        dummy = thisTObj.statsOut[0][0]['test']['obsY']
        testErrs = np.full((runsNum*numFolds,dummy.__len__()),np.nan)
        cvFolds = np.full((runsNum*numFolds,),np.nan)
        num = 0
        for thisRun in np.arange(runsNum):
            # for this run initiate the CV data
            thisCase = thisTObj.statsOut[thisRun]
            for cvNum in np.arange(numFolds):
                obsY = thisCase[cvNum]['test']['obsY']
                modY = thisCase[cvNum]['test']['modY']
                testErrs[num,:] = (modY.values - obsY.values).squeeze()
                cvFolds[num] = cvNum
                num += 1
        print('Done!')
        valCols = (np.arange(testErrs.shape[1])+1)*3
        dataOut[key] = pd.DataFrame(data=testErrs, columns=valCols)

        dataOut[key]['site'] = thisTObj.config['site']
        dataOut[key]['hist'] = thisTObj.config['histBool'] 
        dataOut[key]['cvFold'] = cvFolds
        dataOut[key]['case'] = thisTObj.config['saveClass']
        dataOut[key]['units'] = dataOut[key].index

    # now melt the data
    dataOutMelt = pd.concat([thisDF for _,thisDF in dataOut.items()],sort=False)
    dataOutMelt = pd.melt(dataOutMelt, id_vars=['site','hist','cvFold','units','case'])
    return dataOutMelt

################################################################################
################################################################################

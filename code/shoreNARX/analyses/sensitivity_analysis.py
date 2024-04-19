import pandas as pd
import numpy as np
from tqdm import tqdm

################################################################################
################################################################################

def run_sensitivity_analysis(trainObj,nreps,verbose=True):
    """
    Run a sensitivity analysis on the given training object.

    This function performs a sensitivity analysis by shuffling the input variables
    and observing the effect on the model's performance. The performance is measured
    in terms of Root Mean Square Error (RMSE), Pearson's R correlation coefficient, 
    R2 coefficient of determination, and Normalized Mean Square Error (NMSE).

    Parameters:
    trainObj (object): The training object containing the model, input variables,
                       and other related data.

    Returns:
    pandas.DataFrame: A DataFrame containing the performance metrics (RMSE, R, NMSE)
                      for each shuffled input variable, along with the variable name,
                      site, and cross-validation fold number.
    """
    # Load the data details
    runsNum = trainObj.statsOut.__len__()
    # runsNum = 1
    numFolds = trainObj.statsOut[0].__len__()

    numreps = np.arange(nreps)

    # Get the variable names 
    thisIV = trainObj.inputVars

    # setup collect and test values
    num = 0
    sites = []
    varHold = []
    cvFold = []

    perfHold = np.full((runsNum*numFolds*thisIV.__len__()*numreps.__len__(),4),np.nan)

    for thisRun in tqdm(np.arange(runsNum)):       
        # for this run initiate the CV data
        thisCase = trainObj.statsOut[thisRun]
        thisCVdata = trainObj.makeCVdata(
            trainObj.trainCombs[0].__len__()
        )
        thisSite = trainObj.site
        for cvNum in np.arange(numFolds):
            if verbose:
                print('{}..'.format(cvNum+1),end='')
            # setup the model
            # trainOut = dataStore[thisRun][cvNum]
            stateDict = thisCase[cvNum]['model']
            trainObj.restore_model(thisRun,cvNum)
            thisModel = trainObj.model
            
            testpredy = thisModel.predict(
                thisModel.model,
                thisModel.testX,thisModel.testAdd[0],
                **thisModel.settings)
            
            bestRMSE, bestNMSE, bestR, bestR2 = thisModel.calc_performance(testpredy, thisModel.testY)
            if verbose:
                print('Best RMSE: {:.2f}, best NMSE: {:.2f}, best R: {:.2f}, best R2: {:.2f}'.format(bestRMSE, bestNMSE, bestR, bestR2))
            
            # Account for the fact that we have Hsig_0, Hsig_1 etc (we will do them all!)
            if thisModel.settings['hist']:
                # there are often peak variables so we capture all and shuffle Hsig_0...Hsig_n and Hsig_peak_0...Hsig_peak_n
                varLen = int((thisModel.trainX.shape[1]-1)/(thisIV.__len__()-1))
            else:
                varLen = int((thisModel.trainX.shape[1])/thisIV.__len__())
            
            for ii, var in enumerate(thisIV):
                # shuffle the data
                mixInds = np.arange(ii*varLen,(ii+1)*varLen)
                thisTestX = thisModel.testX.copy()
                shuffInds = np.arange(thisTestX.shape[0])
                
                for _ in numreps:
                    # run the model and predict
                    np.random.shuffle(shuffInds)
                    if ii == thisIV.__len__()-1:
                        # for the case of the shoreline passback, turn off history
                        hist_settings = thisModel.settings.copy()
                        hist_settings['hist'] = False 
                        thisTestX[:,-1] = thisTestX[:,-1][shuffInds]
                        thisPred = thisModel.predict(
                            thisModel.model,
                            thisTestX,thisModel.testAdd[0],
                            **hist_settings)
                    else:
                        thisTestX[:,mixInds] = thisTestX[:,mixInds][shuffInds,:]
                        thisPred = thisModel.predict(
                            thisModel.model,
                            thisTestX,thisModel.testAdd[0],
                            **thisModel.settings)
                    # now calc performance for the shuffled data 
                    thisRMSE, thisNMSE, thisR, thisR2 = thisModel.calc_performance(thisPred, thisModel.testY)
                    # and store
                    varHold.append(var)
                    perfHold[num,0] = thisRMSE - bestRMSE
                    perfHold[num,1] = bestR - thisR
                    perfHold[num,2] = thisNMSE - bestNMSE
                    perfHold[num,3] = bestR2 - thisR2

                    sites.append(thisSite)
                    cvFold.append(cvNum+1)
                    num += 1
    #and finally combine
    looDF = pd.DataFrame(data=perfHold, columns=['RMSE','R','NMSE', 'R2'])
    looDF['Variable'] = varHold
    looDF['Site'] = ['Narrabeen' if 'narra' in _ else 'Tairua' for _ in sites]
    looDF['cvFold'] = cvFold

    return looDF

################################################################################
################################################################################
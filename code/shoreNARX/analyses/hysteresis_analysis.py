import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy as sp

################################################################################
################################################################################

def run_hysteresis_analysis(trainObj,tryNum=1):
    '''
    This function takes trained models and runs a hysteresis analysis to view the 
    effect of the previous shoreline position on the output dx
    input:
        trainObj: trained model wrapper object
        tryNum: number of runs on which to conduct the hysteresis 
            analysis for each beach
    '''
    # Load the data details
    runsNum = trainObj.statsOut.__len__()
    numFolds = trainObj.statsOut[0].__len__()

    # testErrs = np.full((runsNum*numFolds,dummy[0]['Y'].__len__()),np.nan)
    
    # setup collect and test values
    num = 0
    sites = []
    varHold = []
    cvFold = []
    testShlVals = np.linspace(-25,125,100)
    storeDF = [] # pd.DataFrame(columns=testShlVals.round(2).tolist()+['site','Hsig'])
    obsStoreDF = [] # pd.DataFrame(columns=['shlPos','dx','site'])

    # inputVars = caseInfo['Combinations'][2][0].__len__()

    # pick a random two from each site
    randNums = np.random.choice(np.arange(runsNum),size=tryNum,replace=False).tolist()
    # randNums = randNums + [int(_ + runsNum/2) for _ in randNums]

    for thisRun in tqdm(randNums):
        # print('\n{}...\n'.format(thisRun))
        
        # for this run initiate the CV data
        thisCase = trainObj.statsOut[thisRun]
        thisCVdata = trainObj.makeCVdata(
            trainObj.trainCombs[0].__len__()
        )
        thisSite = trainObj.site
        # get the min max shoreline across all folds
        allMaxShl = [np.nanmax(_['Y']) for _ in thisCVdata]
        allMinShl = [np.nanmin(_['Y']) for _ in thisCVdata]
        [minShl, maxShl] = [np.nanmin(allMinShl),np.nanmax(allMaxShl)]
        
        #get the actual data
        if thisSite not in sites:
            sites.append(thisSite)
            for _ in thisCVdata:
                dummyDF = pd.DataFrame(columns=['shlPos','dx','site'])
                dummyDF['shlPos'] = ((_['Add'] - minShl) * 100) / (maxShl - minShl)
                dummyDF['dx'] = _['Y']-_['Add']
                dummyDF.loc[dummyDF['dx']==0,'dx'] = np.nan
                dummyDF['site'] = [thisSite]*_['Add'].shape[0]
                obsStoreDF.append(dummyDF)
        
        for cvNum in np.arange(numFolds):
            # setup the model
            # trainOut = dataStore[thisRun][cvNum]
            stateDict = thisCase[cvNum]['model']
            trainObj.restore_model(thisRun,cvNum)
            thisModel = trainObj.model
            
            # setup the train test split
            unNormedData = thisModel.scalerX.inverse_transform(thisModel.testX)
            
            # get the forcing data
            forceLen = int((unNormedData.shape[1] - 1) / trainObj.inputVars.__len__())
            assert 'Hsig' in trainObj.inputVars[0]
            thisHsig = unNormedData[:,:forceLen]
            
            # get the beach limits - and get dummy values to run through norm transform
            shl_rng = np.array([minShl, maxShl])
            # run extending the range outside of the training data
            shl_rng_X = (testShlVals/100) * (shl_rng[1] - shl_rng[0]) + shl_rng[0]

            limsX = thisModel.trainX[:shl_rng_X.shape[0]].copy()
            limsX[:,-1] = shl_rng_X 
            limsX = thisModel.transform_data(limsX)
            convertedShlVals = limsX[:,-1]

            # for all inputX in test set, run through all shoreline values
            thisTestX = thisModel.transform_data(thisModel.testX.copy())
            thisPred = np.full((thisTestX.shape[0],convertedShlVals.__len__()),np.nan)
            for ii, xAdjust in enumerate(convertedShlVals):
                hist_settings = thisModel.settings.copy()
                hist_settings['hist'] = False
                hist_settings['dx'] = False # output directly from model
                thisTestX[:,-1] = xAdjust
                thisPred[:,ii] = thisModel.predict(thisModel.model,thisTestX,0,**hist_settings)
                # for jj, thisX in enumerate(thisTestX):
                    # thisPred[jj,ii] = thisModel.predict(thisModel.model,thisX,0,**hist_settings)
            
            thisDF = pd.DataFrame(data=thisPred,columns=testShlVals.round(2).tolist())
            
            thisDF['site'] = [thisSite]* thisDF.shape[0]
            thisDF['Hsig'] = thisHsig.mean(axis=1)

            storeDF.append(thisDF)

    storeDF = pd.concat(storeDF,axis=0, ignore_index=True)
    obsStoreDF = pd.concat(obsStoreDF,axis=0, ignore_index=True)
    # storeDF = storeDF.append(thisDF,ignore_index=True)

    return storeDF, obsStoreDF

################################################################################
################################################################################

def run_synthetic_analysis(trainObj,tryNum=1):
    '''
    This function takes trained models and runs a hysteresis analysis to view the 
    effect of the previous shoreline position on the output dx
    input:
        trainObj: trained model wrapper object
        tryNum: number of runs on which to conduct the hysteresis 
            analysis for each beach
    '''
    # Load the data details
    runsNum = trainObj.statsOut.__len__()
    numFolds = trainObj.statsOut[0].__len__()

    # testErrs = np.full((runsNum*numFolds,dummy[0]['Y'].__len__()),np.nan)
    
    # setup collect and test values
    num = 0
    sites = []
    varHold = []
    cvFold = []
    testShlVals = np.linspace(-25,125,100)
    storeDF = [] # pd.DataFrame(columns=testShlVals.round(2).tolist()+['site','Hsig'])
    obsStoreDF = [] # pd.DataFrame(columns=['shlPos','dx','site'])

    # inputVars = caseInfo['Combinations'][2][0].__len__()

    # pick a random two from each site
    randNums = np.random.choice(np.arange(runsNum),size=tryNum,replace=False).tolist()
    # randNums = randNums + [int(_ + runsNum/2) for _ in randNums]

    for thisRun in tqdm(randNums):
        # print('\n{}...\n'.format(thisRun))
        
        # for this run initiate the CV data
        thisCase = trainObj.statsOut[thisRun]
        thisCVdata = trainObj.makeCVdata(
            trainObj.trainCombs[0].__len__()
        )
        thisSite = trainObj.site
        # get the min max shoreline across all folds
        allMaxShl = [np.nanmax(_['Y']) for _ in thisCVdata]
        allMinShl = [np.nanmin(_['Y']) for _ in thisCVdata]
        [minShl, maxShl] = [np.nanmin(allMinShl),np.nanmax(allMaxShl)]
        
        #get the actual data
        if thisSite not in sites:
            sites.append(thisSite)
            for _ in thisCVdata:
                dummyDF = pd.DataFrame(columns=['shlPos','dx','site'])
                dummyDF['shlPos'] = ((_['Add'] - minShl) * 100) / (maxShl - minShl)
                dummyDF['dx'] = _['Y']-_['Add']
                dummyDF['site'] = [thisSite]*_['Add'].shape[0]
                obsStoreDF.append(dummyDF)
        
        for cvNum in np.arange(numFolds):
            # setup the model
            # trainOut = dataStore[thisRun][cvNum]
            stateDict = thisCase[cvNum]['model']
            trainObj.restore_model(thisRun,cvNum)
            thisModel = trainObj.model
            
            # setup the train test split
            unNormedData = thisModel.scalerX.inverse_transform(thisModel.testX)
            
            # get the forcing data
            forceLen = int((unNormedData.shape[1] - 1) / trainObj.inputVars.__len__())
            thisHsig = unNormedData[:,:forceLen]
            
            # get the beach limits - and get dummy values to run through norm transform
            shl_rng = np.array([minShl, maxShl])
            # run extending the range outside of the training data
            shl_rng_X = (testShlVals/100) * (shl_rng[1] - shl_rng[0]) + shl_rng[0]

            limsX = thisModel.trainX[:shl_rng_X.shape[0]].copy()
            limsX[:,-1] = shl_rng_X 
            limsX = thisModel.transform_data(limsX)
            convertedShlVals = limsX[:,-1]

            # for all inputX in test set, run through all shoreline values
            thisTestX = thisModel.transform_data(thisModel.testX.copy())
            thisPred = np.full((thisTestX.shape[0],convertedShlVals.__len__()),np.nan)
            for ii, xAdjust in enumerate(convertedShlVals):
                hist_settings = thisModel.settings.copy()
                hist_settings['hist'] = False
                thisTestX[:,-1] = xAdjust
                for jj, thisX in enumerate(thisTestX):
                    thisPred[jj,ii] = thisModel.predict(thisModel.model,thisX,0,**hist_settings)
            
            thisDF = pd.DataFrame(data=thisPred,columns=testShlVals.round(2).tolist())
            
            thisDF['site'] = [thisSite]* thisDF.shape[0]
            thisDF['Hsig'] = thisHsig.mean(axis=1)

            storeDF.append(thisDF)

    storeDF = pd.concat(storeDF,axis=0, ignore_index=True)
    obsStoreDF = pd.concat(obsStoreDF,axis=0, ignore_index=True)
    # storeDF = storeDF.append(thisDF,ignore_index=True)

    return storeDF, obsStoreDF


################################################################################
################################################################################

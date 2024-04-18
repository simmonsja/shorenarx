import sys, os
from joblib import Parallel, delayed

from shoreNARX import TrainingClass, define_cv_run

def main():
    '''
    This script will run a cross-validation trial for each ensemble member.
    Run this from the base directory with:
        nohup python code/02a_CV_Run_Script.py > nohup_base_narra.out &
    '''
    nEnsembleMembers = 5
    trainingSettings = {
        # Select only one comment out the rest
        # # Narrabeen base
        # 'config': os.path.join('.','config','narra_base_config.json'),
        # # Narrabeen nohist
        # 'config': os.path.join('.','config','narra_nohist_config.json'),
        # # Narrabeen ar1test
        # 'config': os.path.join('.','config','narra_ar1test_config.json'),
        # # Narrabeen datareq
        # 'config': os.path.join('.','config','narra_datareq_config.json'),
        # # Narrabeen sattest 20%
        # 'config': os.path.join('.','config','narra_sattest1_config.json'),
        # # Narrabeen sattest 20% + noise
        # 'config': os.path.join('.','config','narra_sattest2_config.json'),
        
        # Tairua base
        'config': os.path.join('.','config','tairua_base_config.json'),
        # # Tairua nohist
        # 'config': os.path.join('.','config','tairua_nohist_config.json'),
        # # Tairua ar1test
        # 'config': os.path.join('.','config','tairua_ar1test_config.json'),
        # # Tairua datareq
        # 'config': os.path.join('.','config','tairua_datareq_config.json'),
        # # Tairua sattest 20%
        # 'config': os.path.join('.','config','tairua_sattest1_config.json'),
        # # Tairua sattest 20% + noise
        # 'config': os.path.join('.','config','tairua_sattest2_config.json'),
    }

    run_trial_subset = define_cv_run(nEnsembleMembers, trainingSettings, nStart=0)

    # This will create nProcessors x nTrials trials
    nProcessors = 4
    # with multiprocess.Pool(processes=nProcessors) as pool:
    #     pool.map(run_trial_subset, range(nProcessors))
    # There is some issue in my coding that is holding up multiprocessing
    # Rather than sort that out, I am just going to go with joblib's 
    # embarrassingly parallel loops

    # Joblib
    Parallel(n_jobs=nProcessors)(delayed(run_trial_subset)(ii) for ii in range(nProcessors))    

    print('Done!')

if __name__ == '__main__':
    main()
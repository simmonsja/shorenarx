import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

################################################################################
################################################################################
# Sensitivity analysis
################################################################################
################################################################################

def plot_sensitivity_analysis(looDF, case='base', savebool=False):
    '''
    Plot the leave one out sensitivity analysis results
    '''
    fig = plt.figure(figsize=(14,7))
    sns.set_context("poster")
    #sns.set(font_scale=1.8)
    sns.set_style("ticks",{'axes.grid': True})
    ax1 = fig.add_subplot(121)
    #style with sns
    g = sns.barplot(x='Variable', y='RMSE',
                    hue='Site',
                    data=looDF,
                    ax=ax1)
    ax2 = fig.add_subplot(122)
    g = sns.barplot(x='Variable', y='R2',
                    hue='Site',
                    data=looDF,
                    ax=ax2)
    ax2.get_legend().remove()

    # should make this more generic but hey
    assert [_._text for _ in ax1.get_xticklabels()] == ['Hsig','Tp','Wdir','WL_mean','shl_-1_0']

    ax1.set_xticklabels(['$H_{sig}$','$T_{p}$','$W_{dir}$','$WL$','$x_{t-1}$'],rotation=45)
    ax2.set_xticklabels(['$H_{sig}$','$T_{p}$','$W_{dir}$','$WL$','$x_{t-1}$'],rotation=45)

    ax1.set_xlabel('Variable removed')
    ax1.set_ylabel('Test $\Delta$RMSE',labelpad=10)
    ax2.set_xlabel('Variable removed')
    ax2.set_ylabel('Test $\Delta$R2',labelpad=10)

    txt = ax1.text(-0.2, 1.1, 'a)', transform=ax1.transAxes, va='top', fontweight='bold')
    ax2.text(-0.2, 1.1, 'b)', transform=ax2.transAxes, va='top', fontweight='bold')

    plt.subplots_adjust(wspace=0.3)

    if savebool:
        savePath =  os.path.join(
            '.','figures','sensitivity', 'sensitivty_case_{}.pdf'.format(case))
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        plt.savefig(savePath, bbox_extra_artists=(txt,), bbox_inches='tight', dpi=600)

    return

################################################################################
################################################################################
# Hysteresis analysis
################################################################################
################################################################################

def plot_hysteresis(storeDF):
    # seaborn plot
    sns.set_context("talk")
    sns.set_style("ticks",{'axes.grid': True,})
    plot_num = 800 #only plot 100 realisations
    max_wave = storeDF['Hsig'].max()
    plotDF = storeDF.melt(id_vars=['site','Hsig'])
    sites = plotDF.site.unique()
    fig = plt.figure(figsize=(14,9))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    # g = sns.lmplot(
    #     x='variable',
    #     y='value',
    #     col='site',
    #     # palette='bwr',
    #     height=5, aspect=1.25,
    #     markers=["x"],
    #     fit_reg=False,
    #     scatter=False,
    #     legend=False,
    #     data=plotDF,
    #     scatter_kws={
    #         'color':plotDF['Hsig'].values,'s':1,'alpha':0.05,'cmap':'bwr'
    #     }
    # )
    # ax1 = g.axes[0][0]
    scat_size = 2
    scat_alpha = 0.2
    this_site_scatter = storeDF[storeDF['site']==sites[0]].sample(plot_num,axis=0).melt(id_vars=['site','Hsig'])
    sc = ax1.scatter(this_site_scatter.variable.values,
                this_site_scatter.value.values,
                c=this_site_scatter.Hsig.values,s=scat_size,alpha=scat_alpha,
                cmap='bwr',vmin= 0, vmax=max_wave)
    sns.lineplot(x='variable', y='value', data=plotDF[plotDF['site']=='narra'], estimator=np.median, color='xkcd:black',ax=ax1)

    fs_label = 14
    # plot the 25th 50th and 75h percentiles
    for ii, this_perc in enumerate([1,5,25,75,95,99]):
        perc_data = storeDF[storeDF['site']==sites[0]].drop(columns=['site','Hsig']).quantile(this_perc/100,axis=0).to_frame().reset_index(names=['variable']).melt(id_vars='variable',var_name='percentile',value_name='value')
        sns.lineplot(x='variable', y='value', data=perc_data,color='xkcd:black',ax=ax1,linestyle='--')
        # put a small text label on each line outside the axis limits
        ax1.annotate('{}%'.format(this_perc), (perc_data['variable'].values[-1], np.max([perc_data['value'].values[-1],-10])), textcoords="offset points", xytext=(20,-5), ha='left', fontsize=fs_label)

    # perc_var = 'Hsig'
    # # for bin edges in [0,5,25,50,75,95,100]: select data in each bin based on percentile of perc_var
    # perc_edges = storeDF.loc[storeDF['site']==sites[0],'Hsig'].quantile(np.array([0,1,10,50,75,90,95,97.5,99,100])/100).values
    # for ii in np.arange(perc_edges.__len__()-1):
    #     this_perc = 50
    #     thisData = storeDF.loc[(storeDF['site']==sites[0]) & (storeDF['Hsig']>=perc_edges[ii]) & (storeDF['Hsig']<perc_edges[ii+1])].drop(columns=['site','Hsig']).quantile(this_perc/100,axis=0).to_frame().reset_index(names=['variable']).melt(id_vars='variable',var_name='percentile',value_name='value')
    #     sns.lineplot(x='variable', y='value', data=thisData, color='xkcd:black',ax=ax1,linestyle='--')

    ax1.set_ylim(-10,10)
    ax1.set_xlabel('% of observed $x_{t-1}$ as input')
    ax1.set_ylabel('Predicted dx response (m)')
    ax1.set_title('Narrabeen')
    # ax2 = g.axes[0][1]
    this_site_scatter = this_site_scatter = storeDF[storeDF['site']==sites[1]].sample(plot_num,axis=0).melt(id_vars=['site','Hsig'])
    sc = ax2.scatter(this_site_scatter.variable.values,
                     this_site_scatter.value.values,
                     c=this_site_scatter.Hsig.values,
                     s=scat_size,alpha=scat_alpha,
                     vmin= 0, vmax=max_wave,
                     cmap='bwr')
    sns.lineplot(x='variable', y='value', data=plotDF[plotDF['site']=='tairua'], estimator=np.median, color='xkcd:black',ax=ax2)

    # plot the 25th 50th and 75h percentiles
    for this_perc in [1,5,25,75,95,99]:
        perc_data = storeDF[storeDF['site']==sites[1]].drop(columns=['site','Hsig']).quantile(this_perc/100,axis=0).to_frame().reset_index(names=['variable']).melt(id_vars='variable',var_name='percentile',value_name='value')
        sns.lineplot(x='variable', y='value', data=perc_data,color='xkcd:black',ax=ax2,linestyle='--')
        # put a small text label on each line outside the axis limits
        ax2.annotate('{}%'.format(this_perc), (perc_data['variable'].values[-1], np.max([perc_data['value'].values[-1],-10])), textcoords="offset points", xytext=(20,-5), ha='left', fontsize=fs_label)

    # perc_edges = storeDF.loc[storeDF['site']==sites[1],'Hsig'].quantile(np.array([0,1,10,50,75,90,95,97.5,99,100])/100).values
    # for ii in np.arange(perc_edges.__len__()-1):
    #     this_perc = 50
    #     thisData = storeDF.loc[(storeDF['site']==sites[1]) & (storeDF['Hsig']>=perc_edges[ii]) & (storeDF['Hsig']<perc_edges[ii+1])].drop(columns=['site','Hsig']).quantile(this_perc/100,axis=0).to_frame().reset_index(names=['variable']).melt(id_vars='variable',var_name='percentile',value_name='value')
    #     sns.lineplot(x='variable', y='value', data=thisData, color='xkcd:black',ax=ax2,linestyle='--')


    # sns.despine(top=False, right=False, left=False, bottom=False)

    ax2.set_ylim(-10,10)
    ax2.set_xlabel('% of observed $x_{t-1}$ as input')
    ax2.set_ylabel('')
    ax2.set_title('Tairua')
    plt.subplots_adjust(wspace=0.3)
    # plt.colorbar(sc)
    # place colorbar at the bottom of the plot below the x labels and centred between ax1 and ax2
    cbar = plt.colorbar(sc, ax=[ax1,ax2], orientation='horizontal', pad=0.15, aspect=45, label='$H_{sig}$ (m)')

    if False:
        plt.tight_layout()
        savePath =  os.path.join('..', 'figures', 'response', 'hysteresis_case_' + cvCaseNum.__str__() + '.pdf')
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        plt.savefig(savePath, dpi=600)

################################################################################
################################################################################

def plot_observed_hysteresis(obsStoreDF, storeDF):
    # seaborn plot
    adjobsStoreDF = obsStoreDF.copy()
    plotDF = storeDF.melt(id_vars=['site','Hsig'])
    shoreVals = plotDF['variable'].unique()
    adjobsStoreDF['shlPos'] = np.array([shoreVals[np.argmin(abs(_-shoreVals))] for _ in obsStoreDF['shlPos'].values])

    sns.set_context("talk")
    sns.set_style("ticks",{'axes.grid': True,})
    g = sns.lmplot(x='shlPos', y='dx',
                col='site',
                palette='Set1',
                height=5, aspect=1.25,
                markers=["x"],
                fit_reg=False,
                legend=False,
                data=adjobsStoreDF,
                scatter_kws={'s':10,'alpha':0.5})
    ax1 = g.axes[0][0]
    # sns.regplot(x='shlPos', y='dx', data=adjobsStoreDF[adjobsStoreDF['site']=='narra'], 
    #             scatter=False, lowess=True, ci=None, color='r',ax=ax1)
    sns.lineplot(x='shlPos', y='dx', data=adjobsStoreDF[adjobsStoreDF['site']=='narra'], 
                estimator=np.median, color='r',ax=ax1)
    ax1.set_ylim(-10,10)
    ax1.set_xlim(-25,125)
    ax1.set_xlabel('% of observed $x_{t-1}$ as input')
    ax1.set_ylabel('Observed dx response (m)')
    ax1.set_title('Narrabeen')
    ax2 = g.axes[0][1]
    # sns.regplot(x='shlPos', y='dx', data=adjobsStoreDF[adjobsStoreDF['site']=='tairua'], 
    #             scatter=False, lowess=True, ci=None, color='coral',ax=ax2)
    sns.lineplot(x='shlPos', y='dx', data=adjobsStoreDF[adjobsStoreDF['site']=='tairua'],
                estimator=np.median, color='coral',ax=ax2)
    ax2.set_ylim(-10,10)
    ax2.set_xlim(-25,125)
    ax2.set_xlabel('% of observed $x_{t-1}$ as input')
    ax2.set_ylabel('')
    ax2.set_title('Tairua')

    sns.despine(top=False, right=False, left=False, bottom=False)

################################################################################
################################################################################
# Error window analysis
################################################################################
################################################################################

def plot_raw_errorwindow(dataOutMelt):
    fig = plt.figure(figsize=(14,8))
    ax1 = fig.add_subplot(111)
    #style with sns
    sns.set(font_scale=1.3)
    sns.set_style("ticks")
    g = sns.lineplot(x='variable', y='value',
                    units = 'units',
                    hue='hist',
                    estimator=None,
                    #estimator=np.nanmean,
                    data=dataOutMelt,
                    #ci=100,
                    ax=ax1)
    ax1.set_xlabel('days')
    ax1.set_ylabel('MS Error')


################################################################################
################################################################################

def tsplot(ax, data,prc,**kw):
    x = np.arange(data.shape[1])*3
    est = np.nanmean(data, axis=0)
    sd = np.nanstd(data, axis=0)
    cis = (np.nanpercentile(data, prc, axis=0), np.nanpercentile(data, 100-prc, axis=0))
    ax.fill_between(x,cis[0],cis[1],alpha=0.3)
    ax.plot(x,est,**kw,label='_nolegend_')
    ax.margins(x=0)

def plot_errorwindow_analysis(dataOut):
    sns.set_context("poster")
    #sns.set(font_scale=1.8)
    sns.set_style("ticks",{'axes.grid': True})
    sns.set_palette('Set1')

    fig = plt.figure(figsize=(14,10))
    ax1 = fig.add_subplot(211)
    # dataOut[(dataOut['site']=='narra') & (dataOut['hist']==False)]

    for this_case in dataOut.query('site=="narra"')['case'].unique():
        tsplot(
            ax1,
            dataOut.query(
                'site=="narra" and case=="{}"'.format(this_case)
            )[['units','variable','value']].pivot(
                index=['units'],columns=['variable']
            ).values,
            97.5,
            **{'alpha':0.5}
        )
        # tsplot(ax1,dataOut[(dataOut['site']=='narra') & (dataOut['hist']==False)][['units','variable','value']].pivot(index=['units'],columns=['variable']).values, 97.5,
        #     **{'alpha':0.5})

    ax1.set_title('Narrabeen')

    ax1.set_xlabel('Forecast Days')
    ax1.set_ylabel('Mean Residual (m)\nwith 95% confidence')
    ax1.set_ylim(-35,35)

    ax2 = fig.add_subplot(212)

    for this_case in dataOut.query('site=="tairua"')['case'].unique():
        tsplot(
            ax2,
            dataOut.query(
                'site=="tairua" and case=="{}"'.format(this_case)
            )[['units','variable','value']].pivot(
                index=['units'],columns=['variable']
            ).values,
            97.5,
            **{'alpha':0.5}
        )

    ax2.set_title('Tairua')
    ax2.set_xlabel('Forecast Days')
    ax2.set_ylabel('Mean Residual (m)\nwith 95% confidence')
    ax2.set_ylim(-35,35)

    ax2.legend(
        loc=6, labels=dataOut.query('site=="narra"')['case'].unique(),
        bbox_to_anchor=(1.0,1.25),
    )
    # plt.tight_layout()
    # plt.subplots_adjust(right=0.75)

    # add label a) in top left of ax1 and b) in top left of ax2
    txt = ax1.text(-0.05, 1.2, 'a)', transform=ax1.transAxes, va='top', fontweight='bold')
    ax2.text(-0.05, 1.2, 'b)', transform=ax2.transAxes, va='top', fontweight='bold')
    plt.subplots_adjust(hspace=0.5)

    savePath =  os.path.join(
        '.', 'figures', 'forecast_error',
        'forecast_error_base_nohist.pdf'
    )
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath)

################################################################################
################################################################################
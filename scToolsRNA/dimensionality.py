
importsys
importwarnings
importnumpyasnp
importscipy
importsklearn
importscanpyassc
importmatplotlib.pyplotasplt
importseabornassns
importpandasaspd



#IDENTIFYHIGHLYVARIABLEGENES


defget_min_max_norm(data):

return(data-np.nanmin(data))/(np.nanmax(data)-np.nanmin(data))


defrunningquantile(x,y,p,nBins):
"""calculatethequantileofyinbinsofx"""

ind=np.argsort(x)
x=x[ind]
y=y[ind]

dx=(x[-1]-x[0])/nBins
xOut=np.linspace(x[0]+dx/2,x[-1]-dx/2,nBins)

yOut=np.zeros(xOut.shape)

foriinrange(len(xOut)):
ind=np.nonzero((x>=xOut[i]-dx/2)&(x<xOut[i]+dx/2))[0]
iflen(ind)>0:
yOut[i]=np.percentile(y[ind],p)
else:
ifi>0:
yOut[i]=yOut[i-1]
else:
yOut[i]=np.nan

returnxOut,yOut


defget_vscores(E,min_mean=0,nBins=50,fit_percentile=0.1,error_wt=1):
'''
Calculatev-score(above-Poissonnoisestatistic)forgenesintheinputcountsmatrix
Returnv-scoresandotherstats
'''

ncell=E.shape[0]

mu_gene=E.mean(axis=0).A.squeeze()
gene_ix=np.nonzero(mu_gene>min_mean)[0]
mu_gene=mu_gene[gene_ix]

tmp=E[:,gene_ix]
tmp.data**=2
var_gene=tmp.mean(axis=0).A.squeeze()-mu_gene**2
deltmp
FF_gene=var_gene/mu_gene

data_x=np.log(mu_gene)
data_y=np.log(FF_gene/mu_gene)

x,y=runningquantile(data_x,data_y,fit_percentile,nBins)
x=x[~np.isnan(y)]
y=y[~np.isnan(y)]

defgLog(input):returnnp.log(input[1]*np.exp(-input[0])+input[2])
h,b=np.histogram(np.log(FF_gene[mu_gene>0]),bins=200)
b=b[:-1]+np.diff(b)/2
max_ix=np.argmax(h)
c=np.max((np.exp(b[max_ix]),1))

deferrFun(b2):returnnp.sum(abs(gLog([x,c,b2])-y)**error_wt)
b0=0.1
b=scipy.optimize.fmin(func=errFun,x0=[b0],disp=False)
a=c/(1+b)-1

v_scores=FF_gene/((1+a)*(1+b)+b*mu_gene)
CV_eff=np.sqrt((1+a)*(1+b)-1)
CV_input=np.sqrt(b)

returnv_scores,CV_eff,CV_input,gene_ix,mu_gene,FF_gene,a,b


defget_vscores_adata(adata,norm_counts_per_cell=1e6,min_vscore_pctl=85,min_counts=3,min_cells=3,in_place=True):

'''
Identifieshighlyvariablegenes
Requiresalayerofadatathathasbeenprocessedbytotalcountnormalization(e.g.tpm_nolog)
'''

E=adata.layers['tpm_nolog']

#getvariabilitystatistics
stats={}
stats['vscores'],stats['CV_eff'],stats['CV_input'],stats['gene_ix'],stats['mu_gene'],stats['FF_gene'],stats['a'],stats['b']=get_vscores(E)#gene_ix=genesforwhichvscorescouldbereturned
stats['min_vscore_pctl']=min_vscore_pctl

#indexgenesbasedonvscorespercentile
ix2=stats['vscores']>0#ix2=genesforwhichapositivevscorewasobtained
stats['min_vscore']=np.percentile(stats['vscores'][ix2],min_vscore_pctl)
ix3=(((E[:,stats['gene_ix'][ix2]]>=min_counts).sum(0).A.squeeze()>=min_cells)&(stats['vscores'][ix2]>=stats['min_vscore']))#ix3=genespassingfinalmincells&countsthresholds

#highlyvariablegenes=genespassingall3filteringsteps:gene_ix,ix2,andix3
stats['hv_genes']=adata.var_names[stats['gene_ix'][ix2][ix3]]

ifin_place:

#savegene-levelstatstoadata.var
adata.var['highly_variable']=False
adata.var.loc[stats['hv_genes'],'highly_variable']=True
adata.var['vscore']=np.nan
adata.var.loc[adata.var_names[stats['gene_ix']],'vscore']=stats['vscores']
adata.var['mu_gene']=np.nan
adata.var.loc[adata.var_names[stats['gene_ix']],'mu_gene']=stats['mu_gene']
adata.var['ff_gene']=np.nan
adata.var.loc[adata.var_names[stats['gene_ix']],'ff_gene']=stats['FF_gene']

#savevscoreresultstoadata.uns
adata.uns['vscore_stats']=stats
adata.uns['vscore_stats']['hv_genes']=list(adata.uns['vscore_stats']['hv_genes'])

returnNone

else:

#justreturnvscorestats
returnstats


defget_variable_genes(adata,batch_key=None,filter_method='all',top_n_genes=3000,norm_counts_per_cell=1e6,min_vscore_pctl=85,min_counts=3,min_cells=3,in_place=True):

'''
Filtervariablegenesbasedontheirrepresentationwithinindividualsamplebatches
'''

#computeinitialgenevariabilitystatsusingtheentiredataset
get_vscores_adata(adata,norm_counts_per_cell=norm_counts_per_cell,min_vscore_pctl=min_vscore_pctl,min_counts=min_counts,min_cells=min_cells)

#batchhandling:ifnobatchesareprovidedwearealreadydone
ifbatch_key==None:#
ifin_place==True:
adata.var['vscore']=get_min_max_norm(adata.var['vscore'])
returnadata
else:
returnadata.var['highly_variable']

#ifmultiplebatchesarepresent,thenwewilldeterminevariablegenesindependentlyforeachbatch
else:
batch_ids=np.unique(adata.obs[batch_key])
n_batches=len(batch_ids)

#sethvgenefiltermethod
iffilter_method=='any':
count_thresh=0#>0=keephvgenesidentifiedin1ormorebatches
eliffilter_method=='multiple':
count_thresh=1#>1=onlykeephvgenesidentifiedin2ormorebatches
eliffilter_method=='majority':
count_thresh=n_batches/2#onlykeephvgenesidentifiedin>50%ofbatches
eliffilter_method=='all':
count_thresh=n_batches-1#onlykeephvgenesidentifiedin100%ofbatches
eliffilter_method=='top_n_genes':
min_vscore_pctl=0#returnthetophvgenes(#specifiedby'top_n_genes')rankedbymeanscaledvscore
else:
print('Invalidfilteringmethodprovided!')

#identifyvariablegenesforeachbatchseparately
within_batch_hv_genes=[]
within_batch_vscores=np.full(shape=[adata.shape[1],n_batches],fill_value=np.nan)
forn,binenumerate(batch_ids):
adata_batch=adata[adata.obs[batch_key]==b].copy()
withwarnings.catch_warnings():
warnings.simplefilter('ignore')
vscore_stats=get_vscores_adata(adata_batch,norm_counts_per_cell=norm_counts_per_cell,min_vscore_pctl=min_vscore_pctl,min_counts=min_counts,min_cells=min_cells,in_place=False)
hv_genes_this_batch=list(vscore_stats['hv_genes'])
within_batch_hv_genes.append(hv_genes_this_batch)
within_batch_vscores[:,n]=get_min_max_norm(adata.var['vscore'])#scalevscoresfrom0to1

#aggregatebatchstats
adata.var['vscore']=np.nanmean(within_batch_vscores,axis=1)#returnthemeanofscaledvscoresacrossallbatches
within_batch_hv_genes=[gforgeneinwithin_batch_hv_genesforgingene]
within_batch_hv_genes,hv_batch_count=np.unique(within_batch_hv_genes,return_counts=True)

#performhv_genefiltering
iffilter_methodis'top_n_genes':
hv_genes=adata.var['vscore'].sort_values(ascending=False)[0:top_n_genes].index
else:
hv_genes=within_batch_hv_genes[hv_batch_count>count_thresh]

#updateadata
adata.var['highly_variable']=False
adata.var.loc[hv_genes,'highly_variable']=True

ifin_place:
returnNone
else:
returnadata.var['highly_variable']


defplot_ff(adata,gene_ix=None,color=None):

ifgene_ix==None:
gene_ix=adata.var['highly_variable']
else:
gene_ix=adata.var[gene_ix]

ifcolor==None:
color=np.array(['blue'])

mu_gene=adata.var['mu_gene']
ff_gene=adata.var['ff_gene']
a=adata.uns['vscore_stats']['a']
b=adata.uns['vscore_stats']['b']

x_min=0.5*np.min(mu_gene)
x_max=2*np.max(mu_gene)
xTh=x_min*np.exp(np.log(x_max/x_min)*np.linspace(0,1,100))
yTh=(1+a)*(1+b)+b*xTh
plt.figure(figsize=(6,6))
plt.scatter(np.log10(mu_gene),np.log10(ff_gene),c=np.array(['grey']),alpha=0.3,edgecolors=None,s=4)
plt.scatter(np.log10(mu_gene)[gene_ix],np.log10(ff_gene)[gene_ix],c=color,alpha=0.3,edgecolors=None,s=4)

plt.plot(np.log10(xTh),np.log10(yTh))
plt.xlabel('MeanTranscriptsPerCell(log10)')
plt.ylabel('GeneFanoFactor(log10)')
plt.show()


defplot_vscores(adata,gene_ix=None,color=None):

ifgene_ix==None:
gene_ix=adata.var['highly_variable']
else:
gene_ix=adata.var[gene_ix]

ifcolor==None:
color=np.array(['blue'])

mu_gene=adata.var['mu_gene']
vscores_gene=adata.var['vscore']
a=adata.uns['vscore_stats']['a']
b=adata.uns['vscore_stats']['b']

plt.figure(figsize=(6,6))
plt.scatter(np.log10(mu_gene),np.log10(vscores_gene),c=np.array(['grey']),alpha=0.3,edgecolors=None,s=4)
plt.scatter(np.log10(mu_gene)[gene_ix],np.log10(vscores_gene)[gene_ix],c=color,alpha=0.3,edgecolors=None,s=4)

plt.xlabel('MeanTranscriptsPerCell(log10)')
plt.ylabel('GeneVscores(log10)')
plt.show()


defget_covar_genes(adata,minimum_correlation=0.2,show_hist=True):

#Subsetadatatohighlyvariablegenesxcells(countsmatrixonly)
adata_tmp=sc.AnnData(adata[:,adata.var.highly_variable].X)

#Determineiftheinputmatrixissparse
sparse=False
ifscipy.sparse.issparse(adata_tmp.X):
sparse=True

#Getnncorrelationdistanceforeachhighlyvariablegene
gene_correlation_matrix=1-sparse_corr(adata_tmp.X)
np.fill_diagonal(gene_correlation_matrix,np.inf)
max_neighbor_corr=1-gene_correlation_matrix.min(axis=1)

#filtergeneswhosenearestneighborcorrelationisabovethreshold
ix_keep=np.array(max_neighbor_corr>minimum_correlation,dtype=bool).squeeze()

#Preparearandomizeddatamatrix
adata_tmp_rand=adata_tmp.copy()

ifsparse:
mat=adata_tmp_rand.X.todense()
else:
mat=adata_tmp_rand.X

#randomlypermuteeachrowofthecountsmatrix
forcinrange(mat.shape[1]):
np.random.seed(seed=c)
mat[:,c]=mat[np.random.permutation(mat.shape[0]),c]

ifsparse:
adata_tmp_rand.X=scipy.sparse.csr_matrix(mat)
else:
adata_tmp_rand.X=mat

#Getnncorrelationdistancesforrandomizeddata
gene_correlation_matrix=1-sparse_corr(adata_tmp_rand.X)
np.fill_diagonal(gene_correlation_matrix,np.inf)
max_neighbor_corr_rand=1-gene_correlation_matrix.min(axis=1)

#Plothistogramofcorrelationdistances
plt.figure(figsize=(6,6))
plt.hist(max_neighbor_corr_rand,bins=np.linspace(0,1,100),density=False,alpha=0.5,label='random')
plt.hist(max_neighbor_corr,bins=np.linspace(0,1,100),density=False,alpha=0.5,label='data')
plt.axvline(x=minimum_correlation,color='k',linestyle='--',alpha=0.5,linewidth=1)
plt.xlabel('NearestNeighborCorrelation')
plt.ylabel('Counts')
plt.legend(loc='upperright')
plt.show()

returnadata_tmp.var[ix_keep]




#IDENTIFYSIGNIFICANTPCADIMENSIONS


defget_sig_pcs(adata,n_iter=3,nPCs_test=300,threshold_method='95',show_plots=True,zero_center=True,verbose=True,in_place=True):

#Subsetadatatohighlyvariablegenesxcells(countsmatrixonly)
adata_tmp=sc.AnnData(adata[:,adata.var.highly_variable].X)

#Determineiftheinputmatrixissparse
sparse=False
ifscipy.sparse.issparse(adata_tmp.X):
sparse=True

#Geteigenvaluesfrompcaondatamatrix
ifverbose:
print('PerformingPCAondata')
sc.pp.pca(adata_tmp,n_comps=nPCs_test,zero_center=zero_center)
eig=adata_tmp.uns['pca']['variance']

#Geteigenvaluesfrompcaonrandomlypermuteddatamatrices
ifverbose:
print('PerformingPCAonrandomizeddata')
eig_rand=np.zeros(shape=(n_iter,nPCs_test))
eig_rand_max=[]
n_sig_PCs_trials=[]
forjinrange(n_iter):

ifverboseandn_iter>1:sys.stdout.write('\rIteration%i/%i'%(j+1,n_iter));sys.stdout.flush()

adata_tmp_rand=adata_tmp.copy()

ifsparse:
mat=adata_tmp_rand.X.todense()
else:
mat=adata_tmp_rand.X

#randomlypermuteeachrowofthecountsmatrix
forcinrange(mat.shape[1]):
np.random.seed(seed=j+c)
mat[:,c]=mat[np.random.permutation(mat.shape[0]),c]

ifsparse:
adata_tmp_rand.X=scipy.sparse.csr_matrix(mat)
else:
adata_tmp_rand.X=mat

sc.pp.pca(adata_tmp_rand,n_comps=nPCs_test,zero_center=zero_center)
eig_rand_next=adata_tmp_rand.uns['pca']['variance']
eig_rand[j,:]=eig_rand_next
eig_rand_max.append(np.max(eig_rand_next))
n_sig_PCs_trials.append(np.count_nonzero(eig>np.max(eig_rand_next)))

#Seteigenvaluethresholdingmethod
ifthreshold_method=='95':
method_string='Countingthe#ofPCswitheigenvaluesaboverandomin>95%oftrials'
eig_thresh=np.percentile(eig_rand_max,95)
elifthreshold_method=='median':
method_string='Countingthe#ofPCswitheigenvaluesaboverandomin>50%oftrials'
eig_thresh=np.percentile(eig_rand_max,50)
elifthreshold_method=='all':
method_string='Countingthe#ofPCswitheigenvaluesaboverandomacrossalltrials'
eig_thresh=np.percentile(eig_rand_max,100)

#Determine#ofPCdimensionswitheigenvaluesabovethreshold
n_sig_PCs=np.count_nonzero(eig>eig_thresh)

ifshow_plots:

#Ploteigenvaluehistograms
bins=np.logspace(0,np.log10(np.max(eig)+10),50)
sns.histplot(eig_rand.flatten(),bins=bins,kde=False,alpha=1,label='random',stat='probability',color='orange')#,weights=np.zeros_like(data_rand)+1./len(data_rand))
sns.histplot(eig,bins=bins,kde=False,alpha=0.5,label='data',stat='probability')#,weights=np.zeros_like(data)+1./len(data))
plt.legend(loc='upperright')
plt.axvline(x=eig_thresh,color='k',linestyle='--',alpha=0.5,linewidth=1)
plt.xscale('log')
#plt.yscale('log')
plt.xlabel('Eigenvalue')
plt.ylabel('Frequency')
plt.show()

#Plotscree(eigenvaluesforeachPCdimension)
plt.plot([],label='data',color='#1f77b4',alpha=1)
plt.plot([],label='random',color='#ff7f0e',alpha=1)
plt.plot(eig,alpha=1,color='#1f77b4')
forjinrange(n_iter):
plt.plot(eig_rand[j],alpha=1/n_iter,color='#ff7f0e')
plt.legend(loc='upperright')
plt.axhline(y=eig_thresh,color='k',linestyle='--',alpha=0.5,linewidth=1)
plt.yscale('log')
plt.xlabel('PC#')
plt.ylabel('Eigenvalue')
plt.show()

#PlotnPCsaboverandhistograms
sns.set_context(rc={'patch.linewidth':0.0})
sns.histplot(n_sig_PCs_trials,kde=True,stat='probability',color='#1f77b4')
plt.xlabel('#PCsAboveRandom')
plt.ylabel('Frequency')
plt.xlim([0,nPCs_test])
plt.show()

#Printsummarystatstoscreen
ifverbose:
print()
print(method_string)
print('EigenvalueThreshold=',np.round(eig_thresh,2))
print('#SignificantPCs=',n_sig_PCs)

ifin_place:
adata.uns['n_sig_PCs']=n_sig_PCs
adata.uns['n_sig_PCs_trials']=n_sig_PCs_trials
returnNone

else:
returnn_sig_PCs,n_sig_PCs_trials



#ESTIMATEDIMENSIONALITY


defrun_dim_tests_vscore(adata,batch_key=None,gene_filter_method='multiple',nPCs_test=300,n_trials=3,vpctl_tests=None,verbose=True):

ifvpctl_testsisNone:
vpctl_tests=[99,97.5,95,92.5,90,87.5,85,82.5,80,77.5,75,72.5,70,67.5,65,62.5,60]
#vpctl_tests=[99,95,90,85,80,75,70,65,60]

results_vpctl=[]
results_trial=[]
results_nHVgenes=[]
results_nPCs_each=[]

#Determine#ofsignificantPCdimensionsvsrandomizeddatafordifferentnumbersofhighlyvariablegenes
forn,vpctlinenumerate(vpctl_tests):

ifverbose:sys.stdout.write('\rRunningDimensionalityTest%i/%i'%(n+1,len(vpctl_tests)));sys.stdout.flush()

#Getandfiltervariablegenesforthisvcptl
get_variable_genes(adata,batch_key=batch_key,filter_method=gene_filter_method,min_vscore_pctl=vpctl)

#nPCdimensionstestedcannotexceedthe#ofvariablegenes;adjustnPCs_testifneeded
nPCs_test_use=np.min([nPCs_test,np.sum(adata.var.highly_variable)-1])

#Get#significantPCsforthesevariablegenes&thisvcptl
_,n_sig_PCs_trials=get_sig_pcs(adata,n_iter=n_trials,nPCs_test=nPCs_test_use,show_plots=False,zero_center=True,verbose=False,in_place=False)

#Reportresultsfromeachindependenttrial
fortrialinrange(0,n_trials):
results_vpctl.append(vpctl)
results_trial.append(trial)
results_nHVgenes.append(np.sum(adata.var.highly_variable))
results_nPCs_each.append(n_sig_PCs_trials[trial])

#Exportresultstoadata.uns
results=pd.DataFrame({'vscore_pct':results_vpctl,'trial':results_trial,'n_hv_genes':results_nHVgenes,'n_sig_PCs':results_nPCs_each})
adata.uns['dim_test_results']=results
adata.uns['optim_vscore_pctl']=results.vscore_pct[np.argmax(results.n_sig_PCs)]

#Generatelineplot
plt.figure()
ax=plt.subplot(111)
sns.lineplot(x=results.vscore_pct,y=results.n_sig_PCs)
ymin,ymax=ax.get_ylim()
label_gap=(ymax-ymin)/50
ix=results.trial==0
[ax.text(x,ymax+label_gap,name,rotation=90,horizontalalignment='center')forx,y,nameinzip(results[ix].vscore_pct,results[ix].n_sig_PCs,results[ix].n_hv_genes)]
plt.ylabel('#PCsAboveRandom')
plt.xlabel('vscorePercentile')
ax.invert_xaxis()
plt.show()


defrun_dim_tests_scanpy(adata,batch_key=None,nPCs_test=300,n_trials=3,min_disp_tests=None,verbose=True):

ifmin_disp_testsisNone:
min_disp_tests=[0.5,0.6,0.7,0.8,0.9,1,1.2,1.4,1.6,1.8,2]


results_min_disp=[]
results_trial=[]
results_nHVgenes=[]
results_nPCs_each=[]

#Determine#ofsignificantPCdimensionsvsrandomizeddatafordifferentnumbersofhighlyvariablegenes
forn,min_dispinenumerate(min_disp_tests):

ifverbose:sys.stdout.write('\rRunningDimensionalityTest%i/%i'%(n+1,len(min_disp_tests)));sys.stdout.flush()

#Getandfiltervariablegenesforthisvcptl
sc.pp.highly_variable_genes(adata,batch_key=batch_key,layer='tpm',min_mean=0.05,max_mean=10,n_bins=50,min_disp=min_disp)

#nPCdimensionstestedcannotexceedthe#ofvariablegenes;adjustnPCs_testifneeded
nPCs_test_use=np.min([nPCs_test,np.sum(adata.var.highly_variable)-1])

#Get#significantPCsforthesevariablegenes&thisvcptl
_,n_sig_PCs_trials=get_significant_pcs(adata,n_iter=n_trials,nPCs_test=nPCs_test_use,show_plots=False,zero_center=True,verbose=False,in_place=False)

#Reportresultsfromeachindependenttrial
fortrialinrange(0,n_trials):
results_min_disp.append(min_disp)
results_trial.append(trial)
results_nHVgenes.append(np.sum(adata.var.highly_variable))
results_nPCs_each.append(n_sig_PCs_trials[trial])

#Exportresultstoadata.uns
results=pd.DataFrame({'min_disp':results_min_disp,'trial':results_trial,'n_hv_genes':results_nHVgenes,'n_sig_PCs':results_nPCs_each})
adata.uns['dim_test_results']=results

#Generatelineplot
plt.figure()
ax=plt.subplot(111)
sns.lineplot(x=results.min_disp,y=results.n_sig_PCs)
ymin,ymax=ax.get_ylim()
label_gap=(ymax-ymin)/50
ix=results.trial==0
[ax.text(x,ymax+label_gap,name,rotation=90,horizontalalignment='center')forx,y,nameinzip(results[ix].min_disp,results[ix].n_sig_PCs,results[ix].n_hv_genes)]
plt.ylabel('#PCsAboveRandom')
plt.xlabel('MinimumNormalizedDispersion')
ax.invert_xaxis()
plt.show()





#LEGACYALIASES

get_significant_pcs=get_sig_pcs


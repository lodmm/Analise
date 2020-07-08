import pickle
import numpy as numpy 
import pandas as pd
import matplotlib.pyplot as plt

with open('models.pickle','rb') as f:
	models = pickle.load(f)

with open('sets.pickle','rb') as f:
	sets = pickle.load(f)

vars = ['etap_IPCHI2_OWNPV', 'etap_FDCHI2_OWNPV', 'etap_VCHI2PERDOF', 'etap_PT', 'Ds_IPCHI2_OWNPV', 'Ds_FDCHI2_OWNPV', 'Ds_VCHI2PERDOF', 'Ds_PT', 'pi_PT', 'pi_IPCHI2_OWNPV', 'pip_eta_PT', 'pip_eta_IPCHI2_OWNPV', 'pim_eta_PT', 'pim_eta_IPCHI2_OWNPV', 'mup_PT', 'mup_IPCHI2_OWNPV', 'mum_PT', 'mum_IPCHI2_OWNPV']
correlation = ['etap_M','Ds_M']

model = models[0]
set_aux = sets[0]
etap = set_aux['X_test']['etap_M'].to_list()
ds = set_aux['X_test']['Ds_M'].to_list()
predictions = {'AdaBoost':model['pred1'],'etap_M':etap, 'Ds_M': ds}
y_test = set_aux['y_test']
signal_d = {'AdaBoost':[],'etap_M':[], 'Ds_M': []}
background_d = {'AdaBoost':[],'etap_M':[], 'Ds_M': []}
for i in range(0,len(y_test)):
	y = y_test.iloc[i]
	if y == 0:
	#store background events
		for x in background_d.keys():
			background_d[x].append(predictions[x][i])
	if y == 1:
		for x in signal_d.keys():
			signal_d[x].append(predictions[x][i])

mass = set_aux['X_test']['Ds_M']
maxMass = max(mass)
minMass = min(mass)
signal_d = pd.DataFrame(signal_d)
background_d = pd.DataFrame(background_d)
aux_s = dict()
aux_b = dict()
aux_b_e = dict()
aux_s_e = dict()
for i in range(int(minMass),int(maxMass) - 20,20):
	aux_s['range'+str(i)+'-'+str(i+20)] = signal_d[(signal_d['Ds_M']>i) & (signal_d['Ds_M']<i+20)]
	aux_b['range'+str(i)+'-'+str(i+20)] = background_d[(background_d['Ds_M']>i) & (background_d['Ds_M']<i+20)]
mass = set_aux['X_test']['etap_M']
maxMass = max(mass)
minMass = min(mass)
for i in range(int(minMass),int(maxMass) - 10,10):
	aux_s_e['range'+str(i)+'-'+str(i+10)] = signal_d[(signal_d['etap_M']>i) & (signal_d['etap_M']<i+10)]	
	aux_b_e['range'+str(i)+'-'+str(i+10)] = background_d[(background_d['etap_M']>i) & (background_d['etap_M']<i+10)]		
fig, axs = plt.subplots(4, 6, figsize=(15, 6), constrained_layout=True, facecolor='w', edgecolor='k')
i = 0

axs = axs.ravel()

for m in aux_s.keys():
	if not bool(m) :
		continue
	for k in signal_d.keys():
		if k != 'Ds_M' and k != 'etap_M':			
			axs[i].hist(aux_s[m][k],histtype='step',label=k+" Ds_M")
			axs[i].hist(aux_b[m][k],histtype='step',label=k+" Ds_M background")
			axs[i].set_title(m)
			handles, labels = axs[i].get_legend_handles_labels()			
	i+=1
fig.legend(handles, labels, loc='lower right')
fig.savefig("Ds_M.png")		
plt.show()			
fig, axs = plt.subplots(4, 5, figsize=(15, 6), constrained_layout=True, facecolor='w', edgecolor='k')
i = 0
axs = axs.ravel()
for m in aux_s_e.keys():
	if not bool(m) :
		continue
	for k in signal_d.keys():
		if k != 'Ds_M' and k != 'etap_M':			
			axs[i].hist(aux_s_e[m][k],histtype='step',label=k+" etap_M")
			axs[i].hist(aux_b_e[m][k],histtype='step',label=k+" etap_M background")
			axs[i].set_title(m)
			handles, labels = axs[i].get_legend_handles_labels()			
	i+=1	
fig.legend(handles, labels, loc='lower right')
fig.savefig("etap_M.png")		
plt.show()		

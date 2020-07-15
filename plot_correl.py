import pickle
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

with open('models.pickle','rb') as f:
	models = pickle.load(f)

with open('sets.pickle','rb') as f:
	sets = pickle.load(f)
vars = ['etap_IPCHI2_OWNPV', 'etap_FDCHI2_OWNPV', 'etap_VCHI2PERDOF', 'etap_PT', 'Ds_IPCHI2_OWNPV', 'Ds_FDCHI2_OWNPV', 'Ds_VCHI2PERDOF', 'Ds_PT', 'pi_PT', 'pi_IPCHI2_OWNPV', 'pip_eta_PT', 'pip_eta_IPCHI2_OWNPV', 'pim_eta_PT', 'pim_eta_IPCHI2_OWNPV', 'mup_PT', 'mup_IPCHI2_OWNPV', 'mum_PT', 'mum_IPCHI2_OWNPV']
correlation = ['etap_M','Ds_M']

def bins_labels(bins, **kwargs):
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
    plt.xlim(bins[0], bins[-1])

#Get fold 0 
set_aux = sets[0]
model = models[0]

etap = set_aux['X_test']['etap_M'].to_list()
ds = set_aux['X_test']['Ds_M'].to_list()
predictions = {'AdaBoost':model['pred'],'etap_M':etap, 'Ds_M': ds}
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
for i in range(int(minMass),int(maxMass) - 100,100):
	aux_s['range'+str(i)+'-'+str(i+100)] = signal_d[(signal_d['Ds_M']>i) & (signal_d['Ds_M']<i+100)]
	aux_b['range'+str(i)+'-'+str(i+100)] = background_d[(background_d['Ds_M']>i) & (background_d['Ds_M']<i+100)]
mass = set_aux['X_test']['etap_M']
maxMass = max(mass)
minMass = min(mass)
for i in range(int(minMass),int(maxMass) - 50,50):
	aux_s_e['range'+str(i)+'-'+str(i+50)] = signal_d[(signal_d['etap_M']>i) & (signal_d['etap_M']<i+50)]	
	aux_b_e['range'+str(i)+'-'+str(i+50)] = background_d[(background_d['etap_M']>i) & (background_d['etap_M']<i+50)]		
#Print DS_m
fig = plt.figure()
plt.style.use('Solarize_Light2')

for k in aux_s.keys():
	plt.title("Signal correlation")
	values, bins, _ = plt.hist(aux_s[k]['AdaBoost'],histtype='step', align='left', rwidth=1, bins=10,label=k)
	#bins_labels(bins, fontsize=20)
	area = sum(np.diff(bins)*values)
	print(area)
plt.legend(loc=4)	
fig.savefig("images/Ds_M_signal.png")		
plt.show()	
fig = plt.figure()
#plt.style.use('ggplot')
for k in aux_b.keys():
	plt.title("Background correlation")
	values, bins, _ = plt.hist(aux_b[k]['AdaBoost'],histtype='step', align='left', rwidth=1,bins=10,label=k)
	#bins_labels(bins, fontsize=20)
	area = sum(np.diff(bins)*values)
	print(area)
plt.legend(loc=4)	
fig.savefig("images/Ds_M_background.png")		
plt.show()	


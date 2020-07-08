import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle 
from sklearn.utils import shuffle
import uproot

with(open("data/folds.pickle",'rb')) as f:
	folds = pickle.load(f)
with(open("data/signal.pickle",'rb')) as f:
	signal = pickle.load(f)	

vars = ['etap_IPCHI2_OWNPV', 'etap_FDCHI2_OWNPV', 'etap_VCHI2PERDOF', 'etap_PT', 'Ds_IPCHI2_OWNPV', 'Ds_FDCHI2_OWNPV', 'Ds_VCHI2PERDOF', 'Ds_PT', 'pi_PT', 'pi_IPCHI2_OWNPV', 'pip_eta_PT', 'pip_eta_IPCHI2_OWNPV', 'pim_eta_PT', 'pim_eta_IPCHI2_OWNPV', 'mup_PT', 'mup_IPCHI2_OWNPV', 'mum_PT', 'mum_IPCHI2_OWNPV']
correlation = ['etap_M','Ds_M']
#Split signal in test and train 
from sklearn.model_selection import train_test_split
signal_train, signal_test, s_y_train, s_y_test = train_test_split(signal.drop("Y",axis=1),signal["Y"],test_size=0.3, random_state=42)
# print(signal_train)
sets = []
set1 = dict()
for i in folds.keys():
	X_train = signal_train.append(folds[i]['train'].drop("Y",axis=1))
	X_test = signal_test.append(folds[i]['test'].drop("Y",axis=1))
	y_train = s_y_train.append(folds[i]['train']['Y'])
	y_test = s_y_test.append(folds[i]['test']['Y'])
	train = X_train
	train['Y'] = y_train
	train.sample(frac=1)
	train = shuffle(train)
	test = X_test
	test['Y'] = y_test
	test.sample(frac=1)
	test = shuffle(train)
	set1["X_train"] = train.drop("Y",axis=1)
	set1["y_train"] = train["Y"]
	set1["X_test"] = test.drop("Y",axis=1)
	set1["y_test"] = test["Y"]
	sets.append(set1)

model = sets[0]
fig = plt.figure()
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier


classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(
        max_depth=9, min_samples_leaf=0.1, max_features='auto'), learning_rate=0.2, n_estimators=200)
i = 0
models = dict()
with open("sets.pickle",'wb') as f:
	pickle.dump(sets,f)
for model in sets:

	classifier.fit(model['X_train'][vars],model['y_train'])

	pred = classifier.predict_proba(model['X_test'][vars])[:,1]
	pred1 = classifier.predict(model['X_test'][vars])

	from sklearn.metrics import classification_report
	from sklearn.metrics import confusion_matrix

	print(confusion_matrix(model['y_test'], pred1))
	print(classification_report(model['y_test'], pred1))	 

	import sklearn.metrics as metrics
	from sklearn.metrics import precision_recall_curve, roc_curve,plot_roc_curve
	fpr, tpr, _ = metrics.roc_curve(model['y_test'], pred)
	auc = metrics.roc_auc_score(model['y_test'], pred)
	pickle.dump(classifier,open('resultados/adaboost'+str(i)+'.pickle','wb'))
	plt.plot(tpr,1-fpr,label="Fold "+str(i)+'Auc:'+str(auc))
	aux = dict()
	aux['pred1'] = pred1
	aux['fpr'] = fpr
	aux['tpr'] = tpr
	aux['pred'] = pred
	aux['y_test'] = model['y_test']
	models[i] = aux
	#metrics.plot_roc_curve(classifier, model['X_test'][vars], model['y_test'])
	i = i + 1
file = "resultados/tmvasol.root"
tfile = uproot.open(file)
result = tfile.get("dataset")
sol = result.get("Method_BDT")
sol2 = result.get("Method_MLP")
mlp = sol2.get("MLP")
bdtf = sol.get("BDTF")
bdt = sol.get("BDT")
mlp_curve = mlp.get(b'MVA_MLP_rejBvsS;1').pandas()
bdtf_curve = bdtf.get(b'MVA_BDTF_rejBvsS;1').pandas()
bdt_curve = bdt.get(b'MVA_BDT_rejBvsS;1').pandas()
size = len(bdtf_curve['count'].tolist())
with open('models.pickle','wb') as f:
	pickle.dump(models,f)
plt.plot(np.linspace(0,1,size),bdt_curve['count'].tolist(),label="BDT")
plt.plot(np.linspace(0,1,size),bdtf_curve['count'].tolist(),label="BDTF")
plt.plot(np.linspace(0,1,size),mlp_curve['count'].tolist(),label="MLP")
plt.title('Receiver Operating Characteristic')
plt.ylabel('True Positive Rate')
plt.xlabel('1 - False Positive Rate')
plt.legend(loc=4)	
plt.savefig('images/adaboost_1.png')
plt.show()	
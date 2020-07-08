import sys
import ROOT
from ROOT import gSystem, gROOT, gApplication, TFile, TTree, TCut, TMVA


signalFile = "data/signal.root"
backgroundFile = "data/background.root"
outFileName = "resultados/tmva_1.root"

vars = ['etap_IPCHI2_OWNPV', 'etap_FDCHI2_OWNPV', 'etap_VCHI2PERDOF', 'etap_PT', 'Ds_IPCHI2_OWNPV', 'Ds_FDCHI2_OWNPV', 'Ds_VCHI2PERDOF', 'Ds_PT', 'pi_PT', 'pi_IPCHI2_OWNPV', 'pip_eta_PT', 'pip_eta_IPCHI2_OWNPV', 'pim_eta_PT', 'pim_eta_IPCHI2_OWNPV', 'mup_PT', 'mup_IPCHI2_OWNPV', 'mum_PT', 'mum_IPCHI2_OWNPV']
spectators = ['etap_M','Ds_M']

signalF = ROOT.TFile.Open( signalFile ," READ ")
backF = ROOT.TFile.Open(backgroundFile,"READ")


treeS = signalF.Get("DecayTree")
treeB = backF.Get("DecayTree")

ROOT.TMVA.Tools.Instance()
fout = ROOT.TFile(outFileName,"RECREATE")

factory = ROOT.TMVA.Factory("TMVAClassification", fout,
                            ":".join([
                                "!V",
                                "!Silent",
                                "Color",
                                "DrawProgressBar",
                                "Transformations=I;",
                                "AnalysisType=Classification"]
                                     ))
factory.SetVerbose( True )
signalWeight     = 1.0
backgroundWeight = 1.0
dataloader = TMVA.DataLoader("dataset")		
s=""
for i in vars:
	dataloader.AddVariable( i, 'F' )
	s = s + ("("+i+">0)&&")

dataloader.AddSpectator("Ds_M",'F')
dataloader.AddSpectator("etap_M",'F')
dataloader.AddSignalTree(treeS,signalWeight)
dataloader.AddBackgroundTree(treeB,backgroundWeight)

sigCut = ROOT.TCut("")
bgCut = ROOT.TCut("")		

dataloader.PrepareTrainingAndTestTree(sigCut,   # signal events
                                   bgCut,    # background events
                                   ":".join([
                                        "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random:NormMode=NumEvents:!V"
                                       ]))		

# //Boosted Decision Trees
factory.BookMethod(dataloader,TMVA.Types.kBDT, "BDT",
                   "!V:NTrees=500:MinNodeSize=2.5%:MaxDepth=9:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" );

# //Multi-Layer Perceptron (Neural Network)
factory.BookMethod(dataloader, TMVA.Types.kMLP, "MLP",
                   "!H:!V:NeuronType=tanh:VarTransform=N:NCycles=100:HiddenLayers=N+5:TestRate=5:!UseRegulator" );

factory.BookMethod( dataloader, TMVA.Types.kBDT, "BDTF",
                           "!H:!V:NTrees=1000:MinNodeSize=2.5%:UseFisherCuts:MaxDepth=9:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=50" )		
						   
							
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()		


fout.Close()			   
print "=== wrote root file %s\n" % outFileName
print "=== TMVAClassification is done!\n"						   
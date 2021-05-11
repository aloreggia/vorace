
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import random
import math
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Dense, Input
from sklearn import svm
from sklearn.base import clone
import keras.backend as K
from keras.models import Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from corankco.dataset import Dataset
from corankco.scoringscheme import ScoringScheme
from corankco.kemrankagg import KemRankAgg
from corankco.algorithms.enumeration import Algorithm
from keras.utils import to_categorical

from itertools import combinations, permutations


class Vorace_agent:

	initialState=None
	classifier = None
	history = None
	callbacks_list= None
	epochs=40
	batch_size = 0
	

	def __init__(self):
		self.initialState=None
		self.classifier = None
		self.history = None
		
	
	def __init__(self, typeP, nClasses, inputLayer=None, batch_size=0, callbacks_list = None, n_classifiers=10):
		
		if typeP==6:
			typeP=random.randint(0,5)
		if typeP==3:
			typeP=random.randint(0,2)
		
		self.batch_size=batch_size
		self.callbacks_list=callbacks_list
		#print(typeP)
		if typeP==0:
			self.classifier = Vorace_agent.getModel(nClasses, inputLayer)
			self.classifier = Model(inputLayer, self.classifier)
			if nClasses==2:
				self.classifier.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
			else:
				self.classifier.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
			self.initialState = self.classifier.get_weights()
			
		elif typeP==1:
			if random.randint(0,1)==0:
				self.classifier = DecisionTreeClassifier(criterion = "gini", max_depth=random.randint(5,25), random_state=0)
			else:
				self.classifier = DecisionTreeClassifier(criterion = "entropy", max_depth=random.randint(5,25), random_state=0)
			self.initialState = clone(self.classifier)
			
		elif typeP==2:
			A=math.log(pow(2,-5))
			B=math.log(pow(2,5))

			c_value = math.exp(random.uniform(A,B))
			
			if random.randint(0,1)==0:
				self.classifier=svm.SVC(kernel='rbf', C=c_value,gamma='auto', probability=True)
			else:
				A=3
				B=5

				degree = int(round(random.uniform(A,B)))
				#print("C: {}  DEGREE: {}".format(c_value, degree))
				self.classifier = svm.SVC(kernel='poly', degree=degree, C=c_value,gamma='auto', probability=True)
			self.initialState = clone(self.classifier)
			
		elif typeP==4:
			
			value_lists={'bootstrap': [True, False],
						 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
						 'max_features': ['auto', 'sqrt'],
						 'min_samples_leaf': [1, 2, 4],
						 'min_samples_split': [2, 5, 10],
						 'n_estimators': [10,20,50,100,200]}
						 #'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
			
			params={'bootstrap': random.choice(value_lists['bootstrap']),
						 'max_depth': random.choice(value_lists['max_depth']),
						 'max_features': random.choice(value_lists['max_features']),
						 'min_samples_leaf': random.choice(value_lists['min_samples_leaf']),
						 'min_samples_split': random.choice(value_lists['min_samples_split']),
						 'n_estimators': random.choice(value_lists['n_estimators']),}
			
			self.classifier = RandomForestClassifier(**params)
			self.initialState = clone(self.classifier)
		elif typeP==5:
			self.classifier = xgb.XGBClassifier(max_depth=random.randint(3,25), n_estimators=n_classifiers, subsambple=random.random(), colsample_bytree=random.random())
			self.initialState = clone(self.classifier)
		

	def reset(self):
		#print(type(self.classifier))
		if type(self.classifier)==Model:
			self.classifier.set_weights(self.initialState)
		else:
			self.classifier = clone(self.initialState)
		
	def fit(self,x,y,y_oneHot=None):
		if type(self.classifier)==Model:
			self.history = self.classifier.fit(x, y_oneHot,epochs=self.epochs, batch_size=self.batch_size, shuffle=True, callbacks=self.callbacks_list, verbose=0)
			self.history=self.history.history['acc'][-1]
		else:
			self.classifier.fit(x, y)
			y_pred = self.classifier.predict(x)
			self.history =  metrics.accuracy_score(y,y_pred)

	def predict(self,x):
	
		if type(self.classifier)==Model:
			y_pred = self.classifier.predict(x)
		else:
			y_pred = self.classifier.predict_proba(x)
		
		return y_pred
	
	
	def getModel(nClass, inputLayer, nHLayers=4):

		n= random.randint(2,nHLayers)
		nInput = K.int_shape(inputLayer)[1]

		#A=math.log(nInput)
		#B=math.log(nInput**2)
		A=math.log(16)
		B=math.log(128)

		#print("A:"+str(A))
		#print("B:"+str(B))
		activation=('relu','tanh')

		nNodes = int(round(math.exp(random.uniform(A,B))))
		act_fun = random.randint(0,len(activation)-1)
		#print("nNodes:"+str(nNodes))
		x = Dense(nNodes, activation=activation[act_fun])(inputLayer)
		#print(K.int_shape(inputLayer)[1])

		for i in range(1,n):
			#nNodes = random.randint(nInput*2,nInput**2)
			nNodes = int(round(math.exp(random.uniform(A,B))))
			#print(nNodes)
			act_fun = random.randint(0,len(activation)-1)
			#print(act_fun)
			x = Dense(nNodes, activation=activation[act_fun])(x)

		if nClass==2:
			x = Dense(nClass, activation='sigmoid')(x)
		else:
			x = Dense(nClass, activation='softmax')(x)

		return x
	
class Vorace:
	
	models=[]
	best_index = 0
	weights = []
	fscores = []
	epsilon=0.0
	
	def __init__(self, models):
		self.models = models
		self.best_index=0
		self.weights = []
		self.fscores = []
		self.epsilon=0.0
	
	def __init__(self, n_models, profile_type, nInput, nClasses, batch_size, callbacks_list=None):
		
		inputL = Input(shape=(nInput,))
		self.models = []
		self.weights = []
		self.fscores = []
		self.best_index=0
		self.epsilon=0.0
		self.nClasses=nClasses
		
		for i in range(n_models):
			self.models.append(Vorace_agent(profile_type, nClasses, inputL, batch_size, callbacks_list))
		
	def fit(self,x,y, y_oneHot=None, bagging=False):

		if bagging:
			k_fold = StratifiedKFold(n_splits = len(self.models), shuffle = True, random_state=42)
			
			i=0
			for train, test in k_fold.split(x,y):
			
				tmp_hist=self.models[i].fit(x[train],y[train],y_oneHot[train])
				'''
				Compute accuracy to store weights
				'''
				if type(self.models[i])==Model:
					self.weights.append(tmp_hist.history['acc'][-1])
				else:
					y_pred = self.models[i].predict(x[train])
					#print(y)
					#print(y_pred)
					y_pred=np.argmax(y_pred, axis=1)
					tmp_hist =  metrics.f1_score(y[train],y_pred,average="micro")
					self.weights.append(tmp_hist)

				if self.models[i].history > self.models[self.best_index].history:
					self.best_index=i
				i += 1
		else:
			for i in range(len(self.models)):

				tmp_hist=self.models[i].fit(x,y,y_oneHot)
				'''
				Compute accuracy to store weights
				'''
				if type(self.models[i])==Model:
					self.weights.append(tmp_hist.history['acc'][-1])
				else:
					y_pred = self.models[i].predict(x)
					#print(y)
					#print(y_pred)
					y_pred=np.argmax(y_pred, axis=1)
					tmp_hist =  metrics.f1_score(y,y_pred,average="micro")
					self.weights.append(tmp_hist)

				#print(f"HISTORY {i}: {self.weights}")
				if self.models[i].history > self.models[self.best_index].history:
					self.best_index=i


	
	def reset(self):
		self.weights = []
		self.fscores = []
		for i in range(len(self.models)):
			self.models[i].reset()
	
	def predict(self, x, bestClassifier=False, nClasses=10, voting="Sum", argMax=False, weighted=False, tiebreak="best", epsilon=False):
	
		temp_fscores=None
		if bestClassifier:
			y_pred = self.models[self.best_index].predict(x)
		else:
			if weighted==False:
				y_pred, temp_fscores=self.votingSystem(voting,self.models,x, classes=nClasses, proba=True)
			else:
				y_pred, temp_fscores=self.votingSystem(voting,self.models,x, classes=nClasses, proba=True, weights=self.weights)
			
		
		#print(f"=========== {y_pred} ==========")
		#print(f"=========== {y} ==========")
		if tiebreak is "best":
			best_pred = self.models[self.best_index].predict(x)
			return_y_pred=[]
			if voting is "Plurality":
				self.epsilon = 0.0
				
			for i in range(len(y_pred)):
				pred=y_pred[i]
				'''print("1.")
				print(pred)
				print("2.")
				print(np.argmax(pred))'''
				winner = np.argwhere(pred == np.amax(pred))
				
				#print(winner)
				if len(winner)>1: 
					#print("3.TIEBREAK")
					pred=best_pred[i]
					#print("4.CHECKED")
					#print(pred)
				return_y_pred.append(pred)
				
				
				'''#Per calcolare epsilon bisogna:
				# Punteggio del vincitore con Plurality
				if voting is "Plurality":
					vincitore=np.amax(pred) #Il punteggio più alto
					
					# Se il punteggio è maggiore o uguale alla maggioranza
					#print("Pred: {}\t Score: {} \t Vincitore prof: {} \t Vincitore vero: {} \t N Voters: {} \t Label: {}".format(pred, vincitore,np.argmax(pred), y[i][0], len(self.models), (vincitore >= len(self.models)/2 and np.argmax(pred)==y[i][0])))
					if vincitore >= len(self.models)/2 and np.argmax(pred)==y[i][0]:
						self.epsilon += 1'''
					
				
				
			y_pred=return_y_pred
	
			
			if voting is "Plurality":
				#print("EPSILON: {:5.4f} \t N_SAMPLES: {}".format(self.epsilon,len(y_pred)))
				self.epsilon = self.epsilon / len(y_pred)
				#print("EPSILON: {:5.4f}".format(self.epsilon))

		#print(y_pred)	
		if argMax:
			return np.argmax(y_pred, axis=1),temp_fscores
			
		return y_pred, temp_fscores
	
	def scoringVec(scoring,preferences, weights=None):

		n_candidate=len(preferences[0][0])
		n_model=len(preferences[0])
		if weights==None:
			weights = np.ones(n_model, dtype="f4")

		if scoring is 'Plurality':
			scoring=np.zeros(n_candidate, dtype='i2')
			scoring[0]=1
		if scoring is 'HalfApproval':
			scoring=np.zeros(n_candidate, dtype='i2')
			n=int(n_candidate/2)
			if n_candidate%2 == 1: 
				n += 1
			for i in range(n):
				scoring[i]=1
		if scoring is 'Borda':
			scoring=[i for i in range(n_candidate)]
			scoring.sort(reverse=True)

		#print(scoring)
		'''
		For each sample compute the score given the scoring vector
		'''
		n_samples = len(preferences)
		scores=np.zeros((n_samples,n_candidate), dtype="f4")
		for j in range(n_samples):
			profile=np.zeros((n_model,n_candidate), dtype="i2")

			'''
			Sort each model sort its preferences
			'''
			#for i in range(n_model):
			#	temp=Vorace.sortingPref(preferences[j][i])
			#	profile[i,:]=temp
			profile = np.flip(np.argsort(preferences[j]))
			'''print("Sample")
			print(preferences[j])
			print("Profile")
			print(profile)'''

			'''
			j is the sample
			i is the candidate
			m is the model
			'''
			for i in range(n_candidate):
				#print(scoring[i])
				'''for index in profile[:,i]:
					scores[j,index] += weights[index] * scoring[i]'''
				for m in range(n_model):
					#Get the candidate in ith position
					pos = profile[m,i]
					#Compute the score for each candidate given its position pos in the ranking of mth model
					#print("Sample:{} Candidate:{} Model:{} Pos:{}".format(j,i,m,pos))
					scores[j,pos] += weights[m] * scoring[i]
				
			#print(f"SCORING {scoring} {scores[j]}")

		'''print("Scores")
		print(scores)'''
		#assert True==False
		return scores
	
	def votingSystem(self,scoring, listModels, x, y=[], classes=10, best_index=0, weights=None, proba=False):

		#print(y)
		#y_val = np.argmax(y, axis=1)
		if  y!=[]: y_val = np.reshape(y,(len(y),1))
		scores=[]
		if  y!=[]: fscores=[]
		#preferences=np.zeros(l)
		#print(x.shape)
		#print(self.models)
		preferences = np.zeros((x.shape[0],len(self.models),classes))
		preferences_ordered = np.zeros((x.shape[0],len(self.models),classes))
		for i in range(len(self.models)):
			'''
			Get prediction for each sample and concatenate them so each row of array as a prediction
			for the same sample for different model
			'''
			'''
			if type(listModels[i])==Model:
				y_pred = listModels[i].predict(x)
			else:
				y_pred = listModels[i].predict_proba(x)
				#print(listModels[i].classes_)
				#print(y_pred)
			'''
			y_pred = self.models[i].predict(x)
			preferences[:,i]=y_pred
			#preferences_ordered[:,i] = np.flip(np.argsort(y_pred))
			#print(y_pred)

			y_pred = np.argmax(y_pred, axis=1)
			y_pred = np.reshape(y_pred,(len(y_pred),1))

			#print("F-score "+str(i)+"th module: "+ str(f1_score(y_val,y_pred,average="micro")))
			#print(f"F-score {i}th module: {f1_score(y,y_pred,average='micro')}")
			#self.fscores.append(f1_score(y,y_pred,average='micro'))
			#if  y!=[]: fscores.append(f1_score(y,y_pred,average="micro"))

		'''
		Compute voting winner for each sample given rankins from each model
		'''
		if scoring is 'Copeland':
			#print(preferences[0])
			scores=Vorace.copeland(preferences)
		elif scoring is "Sum":
			#print("preferences")
			#print(preferences)
			scores=np.sum(preferences, axis=1)
			#print("scores")
			#print(scores)
		elif scoring is "Mean":
			#print("preferences")
			#print(preferences)
			scores=np.mean(preferences, axis=1)
			#print("scores")
			#print(scores)
		elif scoring is "Kemeny":
			scores=Vorace.rankaggr_brute(preferences)
		else:
			#print("scoring")
			#print(scoring)
			#print("preferences")
			#print(preferences)
			scores=Vorace.scoringVec(scoring,preferences, weights=weights)
			#print(f"SCORING {scoring} {preferences}")

		#print("MEAN fscores:"+str(np.mean(fscores)))
		if y!=[]: return scores, fscores
		return scores, None
	
	def sortingPref(preference):
		#preference = np.array(preference)
		#print(preference)
		#assert len(scoring) >= len(preference)
		'''dict={}
		for i in range(len(preference)):
			dict[i]=preference[i]

		pref = sorted(dict, key=dict.get, reverse=True)
		print(preference)
		print(f"PREF {pref}")
		print(f"SORT {np.argsort(preference)[::-1][:]}")
		exit()'''
		#return pref
		return np.argsort(preference)[::-1][:]

	def copeland(preferences):
		ncandidates=len(preferences[0][0])
		scores=np.zeros((len(preferences),ncandidates), dtype="i2")
		n_model=len(preferences[0])

		'''
		For each preference in the list of preferences
		'''
		for l in range(len(preferences)):
			#print(preferences[l])
			#pairwise = np.zeros(ncandidates, dtype="i2")
			comparison = np.zeros((ncandidates,ncandidates), dtype="i2")
			profile=np.zeros((n_model,ncandidates), dtype="i2")

			#for i in range(n_model):
			#	temp=Vorace.sortingPref(preferences[l][i])
			#	profile[i,:]=temp

			profile = np.flip(np.argsort(preferences[l]))
			#print("profile")
			#print(profile)
			#assert True==False
			#Fisso il primo candidato
			for i in range(0, ncandidates):
				for j in range(0,ncandidates):
					_,i1=np.where(profile==i)
					_,i2=np.where(profile==j)
					comparison[i,j]=np.count_nonzero(np.less(i1,i2))
					'''for p in preferences[l]:
						#Verifico chi vince lo scontro a due
						#p=sortingPref(p)
						#print(p)
						#i1=np.where(p==i)
						i1=p.index(i)
						#i2=np.where(p==j)
						i2=p.index(j)
						#print("i:"+str(i)+" j:"+str(j) + "i1:"+str(i1)+" i2:"+str(i2))
						if i1<i2:
							comparison[i,j] += 1
							#comparison[j,i] -= 1
						else:
							if i1 > i2: #SERVE VERAMENTE?????
								#comparison[i,j] -= 1
								comparison[j,i] += 1'''

			#print(comparison)
			'''
			Count how many pairwise is won or lose by any candidate
			'''
			comparison = comparison - comparison.T
			#print(comparison)
			'''
			Remove negative numbers, i.e. candidates who lose pairwise
			'''
			comparison[comparison <0 ]=0
			'''
			For each candidate, count how many pairwise comparisons are won
			'''
			#print(comparison)
			scores[l] = np.count_nonzero(comparison, axis=1)
			#print(scores[l])
			#assert True==False
			#break

		#print(scores[0])
		return scores

	def rankaggr_brute(preferences):
		'''
		For each sample compute the score given the scoring vector
		'''
		n_candidates=len(preferences[0][0])
		n_model=len(preferences[0])
		n_samples = len(preferences)
		scores=np.zeros((n_samples,n_candidates), dtype="f4")
		#perm = permutations(range(n_candidates))

		#perm = list(perm)

		#print(f"#Model: {n_model} #CAndidates: {n_candidates}")
		#print(f"#Model: {n_model} #SAmples: {n_samples}")
		#print(f"Preferences: {preferences} #Preferences: {len(preferences)}")
		
		for l in range(len(preferences)):
			#rofile=np.zeros((n_model,n_candidates), dtype="i2")
			profile=[]
			#print("********************* PREFERENCES ORIG")
			#print(preferences[l])
			#print("********************* PREFERENCES ORDERED")
			#print(np.unique(preferences[l], axis=1))
			temp_ordered = np.flip(np.unique(preferences[l], axis=1), axis=1)
			#print("********************* PREFERENCES ORDERED INVERSE")
			#print(temp_ordered)
			for i in range(n_model):
				#temp=Vorace.sortingPref(preferences[l][i])
				temp = temp_ordered[i] 
				#print("********************* FIRST PREFERENCES ORDERED INVERSE")
				#print(temp)
				#print(f"********************* PREFERENCES ORIG {l} {i}")
				#print(preferences[l][i])
				#print("********************* INDECES")
				temp=[np.where(preferences[l][i]==temp[j])[0] for j in range(len(temp))]
				#print(temp)
				#exit()
				#print([[x] for x in temp ])
				#profile.append([[x] for x in temp ])
				profile.append(temp)

			#print(len(profile))
			ranks = Dataset(profile)
			sc = ScoringScheme()
			if len(profile[0])>5:
				consensus = KemRankAgg.compute_consensus(ranks, sc, Algorithm.ParCons)
			else:
				consensus = KemRankAgg.compute_consensus(ranks, sc, Algorithm.Exact)

			for c in range(len(consensus.consensus_rankings[0])):
				candidate = consensus.consensus_rankings[0][c][0]
				scores[l][candidate] = n_candidates - c
			
			#print(profile)
			#print(scores[l])
			#exit()
		#return min_dist, best_rank
		return scores
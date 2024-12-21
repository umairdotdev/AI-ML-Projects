## -*- coding: utf-8 -*-
#Modules that are being used
#import sys
import string
import numpy as np
from itertools import product
from functools import reduce
import pandas as pd

#Taking Inputs from the files
inFile1 = None
while inFile1 is None:
    inFile1 = input("\nEnter the name for chain file(eg chain-1.txt): ")
inFile2 = None
while inFile2 is None:
    inFile2 = input("\nEnter the name for emit file(eg emit-1.txt): ")
inFile3 = None
while inFile3 is None:
    inFile3 = input("\nEnter the name for obs file(eg obs-1.txt): ")
outFile = None
while outFile is None:
    outFile = input("\nEnter the name for output file to be saved(eg out-1.txt): ")
#dealing with chain file
def grouping(l,size):
    return [l[i:i+size] for i in range(0, len(l), size)]
#Data manipulation for chain ######################################################
Input = open(inFile1,'r')
Data = []
lineNumber=0
line=Input.readline()
Data.append(line)
while line!='':
    line=Input.readline()
    Data.append(line)
finalData = []
for i in Data:
    finalData.append(i.strip())
finalData = [elem for elem in finalData if elem.strip()]
temp_vector=[]
for i in range(len(finalData)):
    temp_vector.append(finalData[i].split())
length=len(temp_vector)
temp_vector=[list(x) for x in zip(*temp_vector)]
probability_values_XY=[]
for i in temp_vector:
    for j in i:
        probability_values_XY.append(float(j))
alphabet_string = string.printable[36:36+len(temp_vector)]
alphabet_list = list(alphabet_string)
prob_val=grouping(probability_values_XY,len(temp_vector))
A=[] #Contains all the details to use as probabilty transition matrix
for i in range(len(temp_vector)):
    A.append({alphabet_list[q]:prob_val[i][q] for q in range (len(alphabet_list)) })
#######################################################################################
#Data manipulation for emit ######################################################
Input = open(inFile2,'r')
Data = []
lineNumber=0
line=Input.readline()
Data.append(line)
while line!='':
    line=Input.readline()
    Data.append(line)
finalData = []
for i in Data:
    finalData.append(i.strip())
finalData = [elem for elem in finalData if elem.strip()]
probability_values_HT=[]
for i in finalData:
    probability_values_HT.append(float(i))
prob_val=[]
for i in range(len(probability_values_HT)):
    prob_val.append(round(1-probability_values_HT[i],2))
B=[]
for i in range(len(prob_val)):
    B.append({'H':probability_values_HT[i],'T':prob_val[i]})
#######################################################################################
#Data manipulation for obs ######################################################
Input = open(inFile3,'r')
Data = []
lineNumber=0
line=Input.readline()
Data.append(line)
while line!='':
    line=Input.readline()
    Data.append(line)
finalData = []
for i in Data:
    finalData.append(i.strip())
finalData = [elem for elem in finalData if elem.strip()]
temp_list=[]
temp_list.extend(finalData[0].split())
observations=[]
for i in temp_list:
    if i=='1':
        observations.append('H')
    if i=='0':
        observations.append('T')
#######################################################################################

class ProbVector:
    def __init__(self, prob: dict):
        states = prob.keys()
        probs  = prob.values()      
        assert len(states) == len(probs), \
            "The probabilities must match the states."
        assert len(states) == len(set(states)), \
            "The states must be unique."
        assert abs(sum(probs) - 1.0) < 1e-12, \
            "Probabilities must sum up to 1."
        assert len(list(filter(lambda x: 0 <= x <= 1, probs))) == len(probs), \
            "Probabilities must be numbers from [0, 1] interval."
        
        self.states = sorted(prob)
        self.values = np.array(list(map(lambda x: 
            prob[x], self.states))).reshape(1, -1)
        
    @classmethod
    def initialize(cls, states: list):
        size = len(states)
        rand = np.random.rand(size) / (size**2) + 1 / size
        rand /= rand.sum(axis=0)
        return cls(dict(zip(states, rand)))
    
    @classmethod
    def from_numpy(cls, array: np.ndarray, states: list):
        return cls(dict(zip(states, list(array))))

    @property
    def dict(self):
        return {k:v for k, v in zip(self.states, list(self.values.flatten()))}

    @property
    def df(self):
        return pd.DataFrame(self.values, columns=self.states, index=['probability'])

    def __repr__(self):
        return "P({}) = {}.".format(self.states, self.values)

    def __eq__(self, other):
        if not isinstance(other, ProbVector):
            raise NotImplementedError
        if (self.states == other.states) and (self.values == other.values).all():
            return True
        return False

    def __getitem__(self, state: str) -> float:
        if state not in self.states:
            raise ValueError("Requesting unknown probability state from vector.")
        index = self.states.index(state)
        return float(self.values[0, index])

    def __mul__(self, other) -> np.ndarray:
        if isinstance(other, ProbVector):
            return self.values * other.values
        elif isinstance(other, (int, float)):
            return self.values * other
        else:
            NotImplementedError

    def __rmul__(self, other) -> np.ndarray:
        return self.__mul__(other)

    def __matmul__(self, other) -> np.ndarray:
        if isinstance(other, ProbMatrix):
            return self.values @ other.values

    def __truediv__(self, number) -> np.ndarray:
        if not isinstance(number, (int, float)):
            raise NotImplementedError
        x = self.values
        return x / number if number != 0 else x / (number + 1e-12)

    def argmax(self):
        index = self.values.argmax()
        return self.states[index]

class ProbMatrix:
    def __init__(self, prob_vec_dict: dict):
        
        assert len(prob_vec_dict) > 1, \
            "The numebr of input probability vector must be greater than one."
        assert len(set([str(x.states) for x in prob_vec_dict.values()])) == 1, \
            "All internal states of all the vectors must be indentical."
        assert len(prob_vec_dict.keys()) == len(set(prob_vec_dict.keys())), \
            "All observables must be unique."

        self.states      = sorted(prob_vec_dict)
        self.observables = prob_vec_dict[self.states[0]].states
        self.values      = np.stack([prob_vec_dict[x].values \
                           for x in self.states]).squeeze() 

    @classmethod
    def initialize(cls, states: list, observables: list):
        size = len(states)
        rand = np.random.rand(size, len(observables)) \
             / (size**2) + 1 / size
        rand /= rand.sum(axis=1).reshape(-1, 1)
        aggr = [dict(zip(observables, rand[i, :])) for i in range(len(states))]
        prvec = [ProbVector(x) for x in aggr]
        return cls(dict(zip(states, prvec)))

    @classmethod
    def from_numpy(cls, array: 
                  np.ndarray, 
                  states: list, 
                  observables: list):
        p_vecs = [ProbVector(dict(zip(observables, x))) \
                  for x in array]
        return cls(dict(zip(states, p_vecs)))

    @property
    def dict(self):
        return self.df.to_dict()

    @property
    def df(self):
        return pd.DataFrame(self.values, 
               columns=self.observables, index=self.states)

    def __repr__(self):
        return "PM {} states: {} -> obs: {}.".format(
            self.values.shape, self.states, self.observables)

    def __getitem__(self, observable: str) -> np.ndarray:
        if observable not in self.observables:
            raise ValueError("Requesting unknown probability observable from the matrix.")
        index = self.observables.index(observable)
        return self.values[:, index].reshape(-1, 1)
    



class HiddenMarkovCoins:
    def __init__(self, T, E, pi):
        self.T = T  # transmission matrix A
        self.E = E  # emission matrix B
        self.pi = pi
        self.states = pi.states
        self.observables = E.observables
    
    def __repr__(self):
        return "HML states: {} -> observables: {}.".format(
            len(self.states), len(self.observables))
    
    @classmethod
    def initialize(cls, states: list, observables: list):
        T = ProbMatrix.initialize(states, states)
        E = ProbMatrix.initialize(states, observables)
        pi = ProbVector.initialize(states)
        return cls(T, E, pi)
    
    def _create_all_chains(self, chain_length):
        return list(product(*(self.states,) * chain_length))
    
    def score(self, observations: list) -> float:
        def mul(x, y): return x * y
        
        score = 0
        all_chains = self._create_all_chains(len(observations))
        for idx, chain in enumerate(all_chains):
            expanded_chain = list(zip(chain, [self.T.states[0]] + list(chain)))
            expanded_obser = list(zip(observations, chain))
            
            p_observations = list(map(lambda x: self.E.df.loc[x[1], x[0]], expanded_obser))
            p_hidden_state = list(map(lambda x: self.T.df.loc[x[1], x[0]], expanded_chain))
            p_hidden_state[0] = self.pi[chain[0]]
            
            score += reduce(mul, p_observations) * reduce(mul, p_hidden_state)
        return score
    
class HiddenMarkovCoin_FP(HiddenMarkovCoins):
    def _alphas(self, observations: list) -> np.ndarray:
        alphas = np.zeros((len(observations), len(self.states)))
        alphas[0, :] = self.pi.values * self.E[observations[0]].T
        for t in range(1, len(observations)):
            alphas[t, :] = (alphas[t - 1, :].reshape(1, -1) 
                         @ self.T.values) * self.E[observations[t]].T
        return alphas
    
    def score(self, observations: list) -> float:
        alphas = self._alphas(observations)
        return float(alphas[-1].sum())
    
class HiddenMarkovCoin_Simul(HiddenMarkovCoins):
    def run(self, length: int) -> (list, list):
        assert length >= 0, "The chain needs to be a non-negative number."
        s_history = [0] * (length + 1)
        o_history = [0] * (length + 1)
        
        prb = self.pi.values
        obs = prb @ self.E.values
        s_history[0] = np.random.choice(self.states, p=prb.flatten())
        o_history[0] = np.random.choice(self.observables, p=obs.flatten())
        
        for t in range(1, length + 1):
            prb = prb @ self.T.values
            obs = prb @ self.E.values
            s_history[t] = np.random.choice(self.states, p=prb.flatten())
            o_history[t] = np.random.choice(self.observables, p=obs.flatten())
        
        return o_history, s_history
#Find the result
class HiddenMarkovCoin_Uncover(HiddenMarkovCoin_Simul):
    def _alphas(self, observations: list) -> np.ndarray:
        alphas = np.zeros((len(observations), len(self.states)))
        alphas[0, :] = self.pi.values * self.E[observations[0]].T
        for t in range(1, len(observations)):
            alphas[t, :] = (alphas[t - 1, :].reshape(1, -1) @ self.T.values) \
                         * self.E[observations[t]].T
        return alphas
    
    def _betas(self, observations: list) -> np.ndarray:
        betas = np.zeros((len(observations), len(self.states)))
        betas[-1, :] = 1
        for t in range(len(observations) - 2, -1, -1):
            betas[t, :] = (self.T.values @ (self.E[observations[t + 1]] \
                        * betas[t + 1, :].reshape(-1, 1))).reshape(1, -1)
        return betas
    
    def uncover(self, observations: list) -> list:
        alphas = self._alphas(observations)
        betas = self._betas(observations)
        maxargs = (alphas * betas).argmax(axis=1)
        return list(map(lambda x: self.states[x], maxargs))

#Main Code
a=[]
for i in range(length):
    t=ProbVector(A[i])
    a.append(t)
b=[]
for i in range(length):
    t=ProbVector(B[i])
    b.append(t)

A1=[]
A1=ProbMatrix({alphabet_list[q]:a[q] for q in range (len(alphabet_list))})   
B1=[]
B1=ProbMatrix({alphabet_list[q]:b[q] for q in range (len(alphabet_list))})  
pi=ProbVector({alphabet_list[q]:1/length for q in range (len(alphabet_list))})
hcoins = HiddenMarkovCoin_Uncover(A1, B1, pi)
uncovered=hcoins.uncover(observations)


result=[]
for i in uncovered:
    for j in range(len(alphabet_list)):
        if i==alphabet_list[j]:
            
            result.append(str(j))
print('\nThe most likely sequence is:','\n',uncovered,'\n',result)

   
#output file
fout = outFile
fo = open(fout, "w")
for i in uncovered:
    fo.write(i)
fo.write('\n')
for i in result:
    fo.write(i)
fo.close()

hc = HiddenMarkovCoins(A1, B1, pi)        
print("\nProbabilty of observation sequence for {} is {:f}.".format(observations, hc.score(observations)))

all_possible_states = set(alphabet_list)
chain_length = len(observations)  # any int > 0
all_states_chains = list(product(*(all_possible_states,) * chain_length))

df = pd.DataFrame(all_states_chains)
dfp = pd.DataFrame()

for i in range(chain_length):
    dfp['p' + str(i)] = df.apply(lambda x: 
        hcoins.E.df.loc[x[i], observations[i]], axis=1)

scores = dfp.sum(axis=1).sort_values(ascending=False)
df = df.iloc[scores.index]
df['score'] = scores
df.head(10).reset_index() 
print('\nThe following table gives the most likely state given a current state:\n',df)
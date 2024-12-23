# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 19:27:24 2019

@author: Umair Aslam
"""
import numpy as np
import random
import operator
import sys
inFile=sys.argv[1]
outFile=sys.argv[2]
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 20:32:28 2019

@author: Umair Aslam
"""

#Take input from the file
Input = open(inFile,'r')
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

method=finalData[0]
if method=='Q':
        learningRate=finalData[1]
        learningRate=int(learningRate)
        alpha=finalData[2]
        alpha=float(alpha)
        gamma=finalData[3]
        gamma=float(gamma)
        eps=finalData[4]
        eps=float(eps)
        gridDim=[]
        gridDim.extend(finalData[5].split(' '))
        M=gridDim[0]
        M=int(M)
        N=gridDim[1]
        N=int(N)
        numObstacles=finalData[6]
        int_obs=int(numObstacles)
        locObs=[]
        for i in range(int_obs):
            A=finalData[6+i+1]
            locObs.append(A)
        for i in range(int_obs):
            locObs[i]=[int(j) for j in locObs[i].split()]
        numPitfalls=finalData[7+int_obs]    
        int_pitfalls=int(numPitfalls)
        locPits=[]
        for i in range(int_pitfalls):
            B=finalData[7+int_obs+i+1]
            locPits.append(B)
        for i in range(int_pitfalls):
            locPits[i]=[int(j) for j in locPits[i].split()]
        locGoal=finalData[8+int_obs+int_pitfalls]
        locGoal=[int(i) for i in locGoal.split()]
        rewards=[]
        rewards.extend(finalData[7+int_obs+int_pitfalls+2].split(' '))
        r_regStep=rewards[0]
        r_regStep=float(r_regStep)
        r_hitObs=rewards[1]
        r_hitObs=float(r_hitObs)
        r_hitPit=rewards[2]
        r_hitPit=float(r_hitPit)
        r_goal=rewards[3]    
        r_goal=float(r_goal)
        class GridWorld:
            ## Initialise starting data
            def __init__(self):
                # Set information about the gridworld
                self.height = M
                self.width = N
                self.grid = np.zeros(( self.height, self.width))
                self.wall_location=[]
                self.pit_location=[]
                self.goal_location=[]
                self.terminal_states=[]
                
                #initialize all cells as empty
                for i in range(self.width):
                    for j in range(self.height):
                        self.grid[i][j]=r_regStep      
                #Find locations of wall
                for i in locObs:
                    xObs=i[0]-1
                    yObs=i[1]-1
                    self.wall_location.append((yObs,xObs))
                #finding locations of pitfalls
                for i in locPits:
                    xPits=i[0]-1
                    yPits=i[1]-1
                    self.pit_location.append((yPits,xPits))
                #finding goal Location
                self.goal_location.append((locGoal[1]-1,locGoal[0]-1))
                # Set random start location for the agent
                self.current_location = (np.random.randint(0,N),np.random.randint(0,M))
                
                # Set locations for the terminal States
                for i in self.pit_location:
                    self.terminal_states.append(i)
                for j in self.goal_location:
                    self.terminal_states.append(j)
                
                # Set grid rewards 
                for i in self.wall_location:
                    a,b=i
                    self.grid[a,b]=r_hitObs
                for i in self.pit_location:
                    a,b=i
                    self.grid[a,b]=r_hitPit
                for i in self.goal_location:
                    a,b=i
                    self.grid[a,b]=r_goal
        #        self.grid[self.current_location[0],self.current_location[1]]=2
        #        self.grid=self.grid[::-1]
                # Set available actions
                self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            
                
            ## Put methods here:
            def get_available_actions(self):
                return self.actions
            
            def agent_on_map(self):
                grid = np.zeros(( self.height, self.width))
                grid[ self.current_location[0], self.current_location[1]] = 1
                return grid
            
            def get_reward(self, new_location):
                return self.grid[ new_location[0], new_location[1]]
            def is_wall(self,new_location,action):
                if action=='UP':
                    a,b= new_location[0]-1,new_location[1]
                    for i in self.wall_location:
                        if i==(a,b):
                            return True
                if action=='DOWN':
                    a,b= new_location[0]+1,new_location[1]
                    for i in self.wall_location:
                        if i==(a,b):
                            return True
                if action=='LEFT':
                    a,b= new_location[0],new_location[1]-1
                    for i in self.wall_location:
                        if i==(a,b):
                            return True
                if action=='RIGHT':
                    a,b= new_location[0],new_location[1]+1
                    for i in self.wall_location:
                        if i==(a,b):
                            return True
                else:
                    return False
                
            
            def make_step(self, action):
                
                # Store previous location
                last_location = self.current_location
                
                # UP
                if action == 'UP':
                    # If agent is at the top, stay still, collect reward
                    if self.is_wall(last_location,action):
                        a,b=last_location[0]-1,last_location[1]
                        reward=self.grid[a,b]
                    elif last_location[0] == 0:
                        reward = self.get_reward(last_location)
                    
                    else:
                        self.current_location = ( self.current_location[0] - 1, self.current_location[1])
                        reward = self.get_reward(self.current_location)
                
                # DOWN
                elif action == 'DOWN':
                    # If agent is at bottom, stay still, collect reward
                    if self.is_wall(last_location,action):
                        a,b=last_location[0]+1,last_location[1]
                        reward=self.grid[a,b]
                    elif last_location[0] == self.height - 1:
                        reward = self.get_reward(last_location)
                    
                    else:
                        self.current_location = ( self.current_location[0] + 1, self.current_location[1])
                        reward = self.get_reward(self.current_location)
                    
                # LEFT
                elif action == 'LEFT':
                    # If agent is at the left, stay still, collect reward
                    if self.is_wall(last_location,action):
                        a,b=last_location[0],last_location[1]-1
                        reward=self.grid[a,b]
                    elif last_location[1] == 0:
                        reward = self.get_reward(last_location)
                    
                    else:
                        self.current_location = ( self.current_location[0], self.current_location[1] - 1)
                        reward = self.get_reward(self.current_location)
        
                # RIGHT
                elif action == 'RIGHT':
                    # If agent is at the right, stay still, collect reward
                    if self.is_wall(last_location,action):
                        a,b=last_location[0],last_location[1]+1
                        reward=self.grid[a,b]
                    elif last_location[1] == self.width - 1:
                        reward = self.get_reward(last_location)
                    
                    else:
                        self.current_location = ( self.current_location[0], self.current_location[1] + 1)
                        reward = self.get_reward(self.current_location)
                        
                return reward
            
            def check_state(self):
                if self.current_location in self.terminal_states:
                    return 'TERMINAL'
                
        class RandomAgent():        
            # Choose a random action
            def choose_action(self, available_actions):
                return np.random.choice(available_actions)
            
        class Q_Agent():
            # Intialise
            def __init__(self, environment, epsilon, alpha, gamma):
                self.environment = environment
                self.q_table = dict() # Store all Q-values in dictionary of dictionaries 
                for x in range(environment.height): # Loop through all possible grid spaces, create sub-dictionary for each
                    for y in range(environment.width):
                        self.q_table[(x,y)] = {'UP':0, 'DOWN':0, 'LEFT':0, 'RIGHT':0} # Populate sub-dictionary with zero values for possible moves
        
                self.epsilon = epsilon
                self.alpha = alpha
                self.gamma = gamma
                
            def choose_action(self, available_actions):
                """eps-greedy"""
                if np.random.uniform(0,1) < self.epsilon:
                    action = available_actions[np.random.randint(0, len(available_actions))]
                else:
                    q_values_of_state = self.q_table[self.environment.current_location]
                    maxValue = max(q_values_of_state.values())
                    action = np.random.choice([k for k, v in q_values_of_state.items() if v == maxValue])
                
                return action
            
            def learn(self, old_state, reward, new_state, action):
                """Q-value table using Q-learning"""
                q_values_of_state = self.q_table[new_state]
                max_q_value_in_new_state = max(q_values_of_state.values())
                current_q_value = self.q_table[old_state][action]
                
                self.q_table[old_state][action] = (1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * max_q_value_in_new_state)
        
        def play(environment, agent, trials=500, max_steps_per_episode=1000, learn=False):
            reward_per_episode = [] # Initialise performance log
            
            for trial in range(trials): # Run trials
                cumulative_reward = 0 # Initialise values of each game
                step = 0
                game_over = False
                while step < max_steps_per_episode and game_over != True: # Run until max steps or until game is finished
                    old_state = environment.current_location
                    action = agent.choose_action(environment.actions) 
                    reward = environment.make_step(action)
                    new_state = environment.current_location
                    
                    if learn == True: # Update Q-values if learning is specified
                        agent.learn(old_state, reward, new_state, action)
                        
                    cumulative_reward += reward
                    step += 1
                    
                    if environment.check_state() == 'TERMINAL': # If game is in terminal state, game over and start next trial
                        environment.__init__()
                        game_over = True     
                        
                reward_per_episode.append(cumulative_reward) # Append reward for current trial to performance log
                
            return reward_per_episode # Return performance log
        env=GridWorld()
        agentQ=Q_Agent(env,eps,alpha,gamma)
        play(env,agentQ,trials=learningRate,learn=True)
        P=agentQ.q_table
        raw_policy=[]
        for k in sorted(P.keys()):
            raw_policy.append(P[k])
        policyDir=[]
        for i in range(len(raw_policy)):
            policyDir.append(max(raw_policy[i].items(),key=operator.itemgetter(1))[0])
        
        policyDir.reverse()
        for index, item in enumerate(policyDir):
            if item=='UP':
                policyDir[index]='DOWN'
            if item=='DOWN':
                policyDir[index]='UP'
        #coords=[]
        #for i in range(1,N+1):
        #    for j in range(M,0,-1):
        #        a,b=(i,j)
        #        coords.append((a,b))
        coords = [(y,x) for x in range(1,N+1) for y in range(1,M+1)]
        coords.reverse()
        
        for n, i in enumerate(policyDir):
            if i=='UP':
                policyDir[n]=0
            if i=='RIGHT':
                policyDir[n]=1
            if i=='DOWN':
                policyDir[n]=2
            if i=='LEFT':
                policyDir[n]=3
        
        output_dict=dict(zip(coords,policyDir))
        fout = outFile
        fo = open(fout, "w")
        for k, v in sorted(output_dict.items()):
            k=str(k)
            k=k.replace(',','')
            fo.write(k.strip('()') + ' '+ str(v) + '\n')
        fo.close()
if method=='V':
    theeta=finalData[1]
    theeta=float(theeta)
    gamma=finalData[2]
    gamma=float(gamma)
    gridDim=[]
    gridDim=[]
    gridDim.extend(finalData[3].split(' '))
    M=gridDim[0]
    N=gridDim[1]
    M=int(M)
    N=int(N)
    numObstacles=finalData[4]
    int_obs=int(numObstacles)
    locObs=[]
    for i in range(int_obs):
        A=finalData[4+i+1]
        locObs.append(A)
    for i in range(int_obs):
        locObs[i]=[int(j) for j in locObs[i].split()]
    numPitfalls=finalData[5+int_obs]    
    int_pitfalls=int(numPitfalls)
    locPits=[]
    for i in range(int_pitfalls):
        B=finalData[5+int_obs+i+1]
        locPits.append(B)
    for i in range(int_pitfalls):
        locPits[i]=[int(j) for j in locPits[i].split()]
    locGoal=finalData[6+int_obs+int_pitfalls]
    locGoal=[int(i) for i in locGoal.split()]
    rewards=[]
    rewards.extend(finalData[5+int_obs+int_pitfalls+2].split(' '))
    r_regStep=rewards[0]
    r_regStep=float(r_regStep)
    r_hitObs=rewards[1]
    r_hitObs=float(r_hitObs)
    r_hitPit=rewards[2]
    r_hitPit=float(r_hitPit)
    r_goal=rewards[3]    
    r_goal=float(r_goal)
    
    identity = lambda x: x
    
    argmin = min
    argmax = max
    
    
    def argmin_random_tie(seq, key=identity):
        return argmin(shuffled(seq), key=key)
    
    
    def argmax_random_tie(seq, key=identity):
        return argmax(shuffled(seq), key=key)
    
    
    def shuffled(iterable):
        items = list(iterable)
        random.shuffle(items)
        return items
    
    orientations = EAST, NORTH, WEST, SOUTH = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    turns = LEFT, RIGHT = (+1, -1)
    
    
    def turn_heading(heading, inc, headings=orientations):
        return headings[(headings.index(heading) + inc) % len(headings)]
    
    
    def turn_right(heading):
        return turn_heading(heading, RIGHT)
    
    
    def turn_left(heading):
        return turn_heading(heading, LEFT)
    
    def vector_add(a, b):
        return tuple(map(operator.add, a, b))
    
    class MDP:
    
    
        def __init__(self, init, actlist, terminals, transitions={}, states=None, gamma=.9):
            if not (0 < gamma <= 1):
                raise ValueError("An MDP must have 0 < gamma <= 1")
    
            if states:
                self.states = states
            else:
                self.states = set()
            self.init = init
            self.actlist = actlist
            self.terminals = terminals
            self.transitions = transitions
            self.gamma = gamma
            self.reward = {}
    
        def R(self, state):
            """Return a numeric reward for this state."""
            return self.reward[state]
    
        def T(self, state, action):
            """Transition model. From a state and an action, return a list
            of (probability, result-state) pairs."""
            if(self.transitions == {}):
                raise ValueError("Transition model is missing")
            else:
                return self.transitions[state][action]
    
        def actions(self, state):
            if state in self.terminals:
                return [None]
            else:
                return self.actlist
    
    
    class GridMDP(MDP):
    
    
        def __init__(self, grid, terminals, init=(0, 0), gamma=.9):
            grid.reverse()  # because we want row 0 on bottom, not on top
            MDP.__init__(self, init, actlist=orientations,
                         terminals=terminals, gamma=gamma)
            self.grid = grid
            self.rows = len(grid)
            self.cols = len(grid[0])
            for x in range(self.cols):
                for y in range(self.rows):
                    self.reward[x, y] = grid[y][x]
                    if grid[y][x] is not None:
                        self.states.add((x, y))
    
        def T(self, state, action):
            if action is None:
                return [(0.0, state)]
            else:
                return [(0.8, self.go(state, action)),
                        (0.1, self.go(state, turn_right(action))),
                        (0.1, self.go(state, turn_left(action)))]
    
        def go(self, state, direction):
            state1 = vector_add(state, direction)
            return state1 if state1 in self.states else state
    
        def to_grid(self, mapping):
            return list(reversed([[mapping.get((x, y), None)
                                   for x in range(self.cols)]
                                  for y in range(self.rows)]))
        #For debugging purposes
        def to_arrows(self, policy):
            chars = {
                (1, 0): '>', (0, 1): '^', (-1, 0): '<', (0, -1): 'v', None: '.'}
            return self.to_grid({s: chars[a] for (s, a) in policy.items()})
    
    
    def value_iteration(mdp, epsilon=0.001):
        """Solving an MDP by value iteration. [Figure 17.4]"""
        U1 = {s: 0 for s in mdp.states}
        R, T, gamma = mdp.R, mdp.T, mdp.gamma
        while True:
            U = U1.copy()
            delta = 0
            for s in mdp.states:
                U1[s] = R(s) + gamma * max([sum([p * U[s1] for (p, s1) in T(s, a)])
                                            for a in mdp.actions(s)])
                delta = max(delta, abs(U1[s] - U[s]))
            if delta < epsilon * (1 - gamma) / gamma:
                return U
    
    
    def best_policy(mdp, U):
        pi = {}
        for s in mdp.states:
            pi[s] = argmax(mdp.actions(s), key=lambda a: expected_utility(a, s, U, mdp))
        return pi
    
    
    def expected_utility(a, s, U, mdp):
        return sum([p * U[s1] for (p, s1) in mdp.T(s, a)])
   
    
    def Maze(M,N,locObs,locPits,locGoal,r_regStep,r_hitObs,r_hitPit,r_goal):  
        rewardValues=[]
        for row in range(M):
            inner_list=[]
            for col in range(N):
                inner_list.append(r_regStep)
            rewardValues.append(inner_list)
        for i in locObs:
            xObs=i[0]
            yObs=i[1]
            rewardValues[yObs-1][xObs-1]=None
        for i in locPits:
            xPits=i[0]
            yPits=i[1]
            rewardValues[yPits-1][xPits-1]=r_hitPit
        rewardValues[locGoal[1]-1][locGoal[0]-1]=r_goal   
        pit_location=[]
        for i in locPits:
                xPits=i[0]-1
                yPits=i[1]-1
                pit_location.append((xPits,yPits))
        goal_location=[]
        
        goal_location.append((locGoal[0],locGoal[1]))
        terminal_states=[]
        
        for j in goal_location:
            terminal_states.append(j)
        for i in pit_location:
            terminal_states.append(i)
        rewardValues.reverse()
        return rewardValues,terminal_states
    
    
    P,Q=Maze(M,N,locObs,locPits,locGoal,r_regStep,r_hitObs,r_hitPit,r_goal)
    sde=GridMDP(P,terminals=Q,gamma=gamma)
    pi=best_policy(sde,value_iteration(sde,theeta))
    policy_actions=[]
    for k in sorted(pi.keys()):
        policy_actions.append(pi[k])
        
    
    for n, i in enumerate(policy_actions):
        if i==(1,0):
            policy_actions[n]=1
        if i==(0,1):
            policy_actions[n]=0
        if i==(-1,0):
            policy_actions[n]=3
        if i==(0,-1):
            policy_actions[n]=2
        if i==None:
            policy_actions[n]=random.randint(0,3)
    
    
    coords=[]
    for i in range(1,N+1):
        for j in range(1,M+1):
            a,b=(i,j)
            coords.append((a,b))
    #Get coordinates of walls
    wall_location=[]
    for i in locObs:
                xObs=i[0]-1
                yObs=i[1]-1
                wall_location.append((xObs+1,yObs+1))
    
    for i in range(len(coords)):
        for j in wall_location:
            if coords[i]==j:
                policy_actions.insert(i,random.randint(0,3))
    
    #For output file
    output_dict=dict(zip(coords,policy_actions))
    fout = outFile
    fo = open(fout, "w")
    for k, v in sorted(output_dict.items()):
        k=str(k)
        k=k.replace(',','')
        fo.write(k.strip('()') + ' '+ str(v) + '\n')
    fo.close()
    


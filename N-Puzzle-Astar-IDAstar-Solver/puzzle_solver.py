import sys
from copy import deepcopy

class Puzzle:
    def __init__ (self, start, parent):
        self.board = start
        self.parent = parent
        self.f = 0
        self.g = 0
        self.h = 0

    def manhattan(self):
        heuristic=0
        for i in range(0,k):
            for j in range(0,k):
                for i2 in range(0,k):
                    for j2 in range(0,k):
                        if self.board[i][j]==goalState[i2][j2]and self.board[i][j] != 0 :
                            heuristic += abs(i-i2)+abs(j-j2)
        return heuristic

    def goal(self):
        for i in range(k):
            for j in range(k):
                if self.board[i][j] != goalState[i][j]:
                    return False
        return True

    def __eq__(self, other):
        return self.board == other.board

def move_function(curr):
    curr = curr.board
    for i in range(k):
        for j in range(k):
            if curr[i][j] == 0:
                x, y = i, j
                break
    q = []
    # Up
    if x-1 >= 0:
        c = deepcopy(curr)
        c[x][y]=c[x-1][y]
        c[x-1][y]=0
        successor = Puzzle(c, curr)
        q.append(successor)
    # Down
    if x+1 < k:
        c = deepcopy(curr)
        c[x][y]=c[x+1][y]
        c[x+1][y]=0
        successor = Puzzle(c, curr)
        q.append(successor)
    # Left
    if y+1 < k:
        c = deepcopy(curr)
        c[x][y]=c[x][y+1]
        c[x][y+1]=0
        successor = Puzzle(c, curr)
        q.append(successor)
    # Right
    if y-1 >= 0:
        c = deepcopy(curr)
        c[x][y]=c[x][y-1]
        c[x][y-1]=0
        successor = Puzzle(c, curr)
        q.append(successor)
    return q

def LeastFValue(frontier):
    f = frontier[0].f
    index = 0
    for i, item in enumerate(frontier):
        if i == 0: 
            continue
        if(item.f < f):
            f = item.f
            index  = i

    return frontier[index], index

def AStar(start):
    frontier = []
    explored = []
    frontier.append(start)

    while frontier:
        current, index = LeastFValue(frontier)
        if current.goal():
            return current
        frontier.pop(index)
        explored.append(current)

        children = move_function(current)
        for child in children:
            ok = False  # checking in closedList
            for i, item in enumerate(explored):
                if item == child:
                    ok = True
                    break
            if not ok:  # not in closed list
                newG = current.g + 1 
                present = False

                # openList includes move
                for j, item in enumerate(frontier):
                    if item == child:
                        present = True
                        if newG < frontier[j].g:
                            frontier[j].g = newG
                            frontier[j].f = frontier[j].g + frontier[j].h
                            frontier[j].parent = current
                if not present:
                    child.g = newG
                    child.h = child.manhattan()
                    child.f = child.g + child.h
                    child.parent = current
                    frontier.append(child)
        if child.f > M:
            print('FAILURE')
            break
    return None

def LimitedFSearch(start,g,fMax):
    start.f=g+start.manhattan()
    if start.f>fMax:
        return None,start.f
    if start.goal():
        return start, start.f
    minim=M
    X=move_function(start)
    for child in X:
        path.append(child)
        goal,newFMax=LimitedFSearch(child,g+1,fMax)
        if goal is not None:
            return goal,newFMax
        if newFMax<minim:
            minim=newFMax
    return None,minim

def IDAStar(start):
    fMax=start.manhattan()
    if start.goal():
        return start
    while True:
        goalFound,newFMax=LimitedFSearch(start,0,fMax)
        if goalFound is not None:
            break
        if fMax>M:
            return False
        fMax=newFMax
    return goalFound

def FindBlank(cur1):
    tempCur=cur1
    blank=None
    blank=tempCur.index('_')
    return blank

def reshape(state,k):
    blankIndex=FindBlank(state)
    state[blankIndex]=0
    state=[int(x) for x in state]
    state=[state[i:i+k] for i in range(0,len(state),k)]
    return state 

def printResult(state):
    for innerList in state:
        for item in innerList:
            if item==0:
                item='_'
            item=str(item)
            print(item + " ",end='')
        print("")

def is_solvable(puzzle, k):
    # Flatten the puzzle, ignoring the blank (0)
    flat_puzzle = [num for row in puzzle for num in row if num != 0]
    
    # Count inversions
    inversions = 0
    for i in range(len(flat_puzzle)):
        for j in range(i + 1, len(flat_puzzle)):
            if flat_puzzle[i] > flat_puzzle[j]:
                inversions += 1
    
    # Check solvability
    if k % 2 == 1:  # Odd grid
        return inversions % 2 == 0
    else:  # Even grid
        blank_row = next(i for i, row in enumerate(puzzle) if 0 in row)
        blank_row_from_bottom = k - blank_row
        return (blank_row_from_bottom % 2 == 0 and inversions % 2 == 1) or \
               (blank_row_from_bottom % 2 == 1 and inversions % 2 == 0)

# Following code block reads inputs from a file
if len(sys.argv) < 2:
    print("Usage: python puzzle_solver.py <input_file>")
    sys.exit(1)

input_file = sys.argv[1]

with open(input_file, 'r') as f:
    lines = f.readlines()

solverMethod = lines[0].strip()
M = int(lines[1].strip())
k = int(lines[2].strip())

initial = []
for i in range(3, 3 + k):
    initial.append(lines[i].strip().split())

goal = []
for i in range(3 + k, 3 + 2 * k):
    goal.append(lines[i].strip().split())

startStateRaw = sum(initial, [])
goalStateRaw = sum(goal, [])

startState = reshape(startStateRaw, k)
goalState = reshape(goalStateRaw, k)

if not is_solvable(startState, k):
    print("This puzzle is unsolvable.")
    sys.exit(1)

Solution = []
Solution.append(goalState)

if solverMethod=='A*':
    start=Puzzle(startState,None)
    result=AStar(start)
    if(not result):
        pass
    else:
        print("\n")
        print('SUCCESS')
        print("\n")
        t=result.parent
        while t:
            Solution.append(t.board)
            t=t.parent
        Solution.reverse()
        for i in range(len(Solution)):
            printResult(Solution[i])
            print("\n")

path=[]

if solverMethod=='IDA*':
    start=Puzzle(startState,None)
    path.append(start)
    result=IDAStar(start)
    if result is None:
        print("\n")
        print('FAILURE')
        print("\n")
    if result is not None:
        print("\n")
        print('SUCCESS')
        print("\n")
        path.reverse()
        index=0
        sol=[]
        sol.append(goalState)
        while result:
            item=path[index].parent
            sol.append(item)
            for j in range(len(path)):
                if item==path[j].board:
                    index=j
            if path[index]==start:
                break
        sol.reverse()
        for i in range(len(sol)):
            printResult(sol[i])
            print("\n")

import sys
import collections
import numpy as np
import heapq
import time
import numpy as np
global posWalls, posGoals
class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""
    def  __init__(self):
        self.Heap = []
        self.Count = 0
        self.len = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

"""Load puzzles and define the rules of sokoban"""

def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n','') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ': layout[irow][icol] = 0   # free space
            elif layout[irow][icol] == '#': layout[irow][icol] = 1 # wall
            elif layout[irow][icol] == '&': layout[irow][icol] = 2 # player
            elif layout[irow][icol] == 'B': layout[irow][icol] = 3 # box
            elif layout[irow][icol] == '.': layout[irow][icol] = 4 # goal
            elif layout[irow][icol] == 'X': layout[irow][icol] = 5 # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)]) 

    # print(layout)
    return np.array(layout)
def transferToGameState2(layout, player_pos):
    """Transfer the layout of initial puzzle"""
    maxColsNum = max([len(x) for x in layout])
    temp = np.ones((len(layout), maxColsNum))
    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = layout[i][j]

    temp[player_pos[1]][player_pos[0]] = 2
    return temp

def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0]) # e.g. (2, 2)

def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5))) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))

def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1)) # e.g. like those above

def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5))) # e.g. like those above

def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)

def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper(): # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls

def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox: # the move was a push
            action.pop(2) # drop the little letter
        else:
            action.pop(3) # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else: 
            continue     

    return tuple(tuple(x) for x in legalActions) # e.g. ((0, -1, 'l'), (0, 1, 'R'))

def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper(): # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox

def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1), 
                    (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1), 
                    (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls: return True
    return False

"""Implement all approcahes"""

def depthFirstSearch(gameState):
    """Implement depthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = collections.deque([[startState]])
    exploredSet = set()
    actions = [[0]] 
    temp = []
    while frontier:
        node = frontier.pop()
        node_action = actions.pop()
        if isEndState(node[-1][-1]):
            temp += node_action[1:]
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                if isFailed(newPosBox):
                    continue
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])
    return temp

def breadthFirstSearch(gameState):
    """Implement breadthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = collections.deque([[startState]]) # lưu lại các state
    exploredSet = set() # tập đóng hỗ trợ việc expand sau này
    actions = collections.deque([[0]]) # lưu lại các action tương ứng
    temp = []
    ### Chọn 1 node trong frontier  và expand cho đến khi frontier rỗng ###
    while frontier:
        # Chọn node ở đầu hàng đợi (nút trái cùng)
        node = frontier.popleft()
        node_action = actions.popleft()

        # Kiểm tra xem node hiện tại có phải goal state
        # Nếu là goal state thì lưu lại 1 lời giải (start state --> goal state) vào temp
        # Ngược lại break
        if isEndState(node[-1][-1]):
            temp += node_action[1:]
            break

        # Kiểm tra xem node được chọn có nằm trong tập đóng exploredSet chưa
        # Nếu chưa thì expand
        # Nếu đã có thì không cần phải expand
        if node[-1] not in exploredSet:
            # Thêm node vào tập đóng
            exploredSet.add(node[-1])

            # Từ node được thêm, expand theo các hướng trên, dưới, trái, phải
            # Đồng thời cập nhật lại vị trí cho player và box
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)

                # Kiểm tra xem trạng thái mới có bị fail không
                # Nếu fail, chẳng hạn đâm vào tường, thì không thêm vào hàng đợi
                if isFailed(newPosBox):
                    continue

                # Thêm vị trí player và box cùng action tương ứng vào hàng đợi
                frontier.append([(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])
    print(len(temp))
    return temp

def cost(actions):
    """A cost function"""
    return len([x for x in actions if x.islower()])

def uniformCostSearch(gameState):
    """Implement uniformCostSearch approach"""
    start = time.time()
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = PriorityQueue() # lưu lại các state
    frontier.push([startState], 0)
    exploredSet = set() # tập đóng hỗ trợ việc expand sau này
    actions = PriorityQueue() # lưu lại các action tương ứng
    actions.push([0], 0)
    temp = []
    numOfNodes = 1
    ### Chọn 1 node trong frontier  và expand cho đến khi frontier rỗng ###
    while frontier:
        # Chọn node đầu Priority Queue (chi phí thấp nhất)
        node = frontier.pop()
        node_action = actions.pop()

        # Kiểm tra xem node hiện tại có phải goal state
        # Nếu là goal state thì lưu lại 1 lời giải (start state --> goal state) vào temp
        # Ngược lại break
        if isEndState(node[-1][-1]):
            temp += node_action[1:]
            break

        # Kiểm tra xem node được chọn có nằm trong tập đóng exploredSet chưa
        # Nếu chưa thì expand
        # Nếu đã có thì không cần phải expand
        if node[-1] not in exploredSet:
            # Thêm node vào tập đóng
            exploredSet.add(node[-1])
            
            # Từ node được thêm, expand theo các hướng trên, dưới, trái, phải
            # Đồng thời cập nhật lại vị trí cho player và box
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)

                # Kiểm tra xem trạng thái mới có bị fail không
                # Nếu fail, chẳng hạn đâm vào tường, thì không thêm vào hàng đợi
                if isFailed(newPosBox):
                    continue

                numOfNodes += 1

                # Thêm action mới cùng chi phí đến state hiện tại (action_cost)
                # Khi push vào frontier sẽ được sắp xếp
                action_cost = cost(node_action[1:] + [action[-1]])
                frontier.push([(newPosPlayer, newPosBox)], action_cost)
                actions.push(node_action + [action[-1]], action_cost)

    end = time.time()
    print("Time:", 1000 * (end - start))
    print("Length:",len(temp))
    print("Number of Nodes expanded:", numOfNodes)
    return temp

def heuristic1(posPlayer, posBox):
    # print(posPlayer, posBox)
    """A heuristic function to calculate the overall distance between the else boxes and the else goals"""
    distance = 0
    completes = set(posGoals) & set(posBox)
    sortposBox = list(set(posBox).difference(completes))
    sortposGoals = list(set(posGoals).difference(completes))
    for i in range(len(sortposBox)):
        distance += (abs(posPlayer[0] - sortposBox[i][0])) + (abs(posPlayer[1] - sortposBox[i][1])) + (abs(sortposBox[i][0] - sortposGoals[i][0])) + (abs(sortposBox[i][1] - sortposGoals[i][1]))
    return distance

def heuristic(posPlayer, posBox):
    # print(posPlayer, posBox)
    """A heuristic function to calculate the overall distance between the else boxes and the else goals"""
    distance = 0
    completes = set(posGoals) & set(posBox)
    sortposBox = list(set(posBox).difference(completes))
    sortposGoals = list(set(posGoals).difference(completes))
    for i in range(len(sortposBox)):
        distance += (abs(sortposBox[i][0] - sortposGoals[i][0])) + (abs(sortposBox[i][1] - sortposGoals[i][1]))
    return distance

def aStarSearch(gameState):
    """Implement aStarSearch approach"""
    start =  time.time()
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)
    temp = []
    numOfNodes = 1
    start_state = (beginPlayer, beginBox)
    frontier = PriorityQueue() # lưu lại các state
    frontier.push([start_state], heuristic(beginPlayer, beginBox))
    exploredSet = set() # tập đóng hỗ trợ việc expand sau này
    actions = PriorityQueue() # lưu lại các action tương ứng
    actions.push([0], heuristic(beginPlayer, start_state[1]))

    ### Chọn 1 node trong frontier  và expand cho đến khi frontier rỗng ###
    while len(frontier.Heap) > 0:
        node = frontier.pop()
        node_action = actions.pop()

        # Kiểm tra xem node hiện tại có phải goal state
        if isEndState(node[-1][-1]):
            temp += node_action[1:]
            break

        # Kiểm tra xem node được chọn có nằm trong tập đóng exploredSet chưa
        # Nếu chưa thì expand
        # Nếu đã có thì không cần phải expand
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            Cost = cost(node_action[1:])

            # Từ node được thêm, expand theo các hướng trên, dưới, trái, phải
            # Đồng thời cập nhật lại vị trí cho player và box
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)

                # Kiểm tra xem trạng thái mới có bị fail không
                # Nếu fail, chẳng hạn đâm vào tường, thì không thêm vào hàng đợi
                if isFailed(newPosBox):
                    continue

                numOfNodes += 1

                # Tính giá trị cho Heuristic
                Heuristic = heuristic1(newPosPlayer, newPosBox)

                # Thêm action mới
                frontier.push(node + [(newPosPlayer, newPosBox)], Heuristic + Cost) 
                actions.push(node_action + [action[-1]], Heuristic + Cost)

    end =  time.time()
    print("Time:", 1000 * (end - start))
    print("Length:",len(temp))
    print("Number of Nodes expanded:", numOfNodes)
    return temp

"""Read command"""
def readCommand(argv):
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('assets/levels/' + options.sokobanLevels,"r") as f: 
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args

def get_move(layout, player_pos, method):
    time_start = time.time()
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    
    if method == 'dfs':
        result = depthFirstSearch(gameState)
    elif method == 'bfs':        
        result = breadthFirstSearch(gameState)
    elif method == 'ucs':
        result = uniformCostSearch(gameState)
    elif method == 'astar':
        result = aStarSearch(gameState)        
    else:
        raise ValueError('Invalid method.')
    time_end=time.time()
    print('Runtime of %s: %.2f second.' %(method, time_end-time_start))
    print(result)
    return result

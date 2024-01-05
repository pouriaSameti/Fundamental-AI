# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


def heuristic(current_game_state: GameState):
    # next_game_state = current_game_state.generatePacmanSuccessor(action)
    # new_position = next_game_state.getPacmanPosition()
    # new_food = next_game_state.getFood()
    # food_position = new_food.asList()
    # new_ghost_states = next_game_state.getGhostStates()
    # new_scared_times = [ghost_state.scaredTimer for ghost_state in new_ghost_states]
    # score = 0
    
    new_position = current_game_state.getPacmanPosition()
    new_food = current_game_state.getFood()
    food_position = new_food.asList()
    new_ghost_states = current_game_state.getGhostStates()
    new_scared_times = [ghost_state.scaredTimer for ghost_state in new_ghost_states]
    score = 0
    
    food_distances = []
    for food in food_position:
        food_distances.append(util.manhattanDistance(food, new_position))

    for rooh in new_ghost_states:
        if util.manhattanDistance(rooh.getPosition(), new_position) <= 3:
            score -= 30
    
    if len(food_distances) > 0:
        closest_food = min(food_distances)
        score += closest_food

    if not current_game_state.hasFood(new_position[0], new_position[1]):
        score -= 10

    if current_game_state.isLose():
        score -= 500

    for ghostState in new_ghost_states:
        if ghostState.scaredTimer >= 1:
            manhattan_distance = manhattanDistance(new_position, ghostState.getPosition())
            if manhattan_distance == 0:
                score += 100

    return score
    

def scoreEvaluationFunction(currentGameState: GameState):
    # legal_actions = currentGameState.getLegalActions(0)
    # scores = []
    # for action in legal_actions:
    #     scores.append(heuristic(currentGameState,action))
    # return max(scores)
    # return currentGameState.getScore() + heuristic(current_game_state=currentGameState, action='WEST')
    h = heuristic(currentGameState)
    score = currentGameState.getScore()
    print('hueristic', h)
    print('score', score)
    return score + h


class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn="scoreEvaluationFunction", depth="2", time_limit="6"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.time_limit = int(time_limit)


class AIAgent(MultiAgentSearchAgent):

    def getAction(self, gameState: GameState):
        chosen_action = self.minimax(gameState, self.depth, agent_index=self.index, pac_turn=True)[1]
        return chosen_action
        util.raiseNotDefined()
        
    def minimax(self, game_state: GameState, depth: int, agent_index: int, pac_turn = True):
        if depth == 0 or game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state), Directions.STOP
        
        if pac_turn:
            return self.pac_value(game_state, depth, agent_index)
        else: 
            return self.rooh_value(game_state, depth, agent_index)
    
    def pac_value(self, game_state: GameState, depth: int, agent_index: int):
        legal_actions = game_state.getLegalActions(agent_index)
        scores = []
        for action in legal_actions:
            next_game_state = game_state.generateSuccessor(agent_index, action)
            scores.append(self.minimax(next_game_state, depth, agent_index = agent_index + 1, pac_turn=False)[0])
        
        max_score = max(scores)
        max_indexes = [i for i, score in enumerate(scores) if score == max_score]
        chosen_action = legal_actions[random.choice(max_indexes)]
        return max_score, chosen_action
    
    def rooh_value(self, game_state: GameState, depth: int, agent_index: int): 
        legal_actions = game_state.getLegalActions(agent_index)
        scores = []
        for action in legal_actions:
            next_game_state = game_state.generateSuccessor(agent_index, action)
            
            if agent_index == game_state.getNumAgents() - 1: # next turn = pac turn
                scores.append(self.minimax(next_game_state, depth - 1, agent_index=0, pac_turn=True)[0])
            else: 
                scores.append(self.minimax(next_game_state, depth, agent_index=agent_index + 1, pac_turn=False)[0])
        
        min_score = min(scores)
        min_indexes = [i for i, score in enumerate(scores) if score == min_score]
        chosen_action = legal_actions[random.choice(min_indexes)]
        return min_score, chosen_action
    
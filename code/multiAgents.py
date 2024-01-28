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
    
    new_position = current_game_state.getPacmanPosition()
    new_food = current_game_state.getFood()
    food_position = new_food.asList()
    new_ghost_states = current_game_state.getGhostStates()
    score = current_game_state.getScore()
    
    food_distances = []
    for food in food_position:
        food_distances.append(util.manhattanDistance(food, new_position))

    for rooh in new_ghost_states:
        if util.manhattanDistance(rooh.getPosition(), new_position) <= 1 and util.manhattanDistance(rooh.getPosition(), new_position) != 0:
            score -= 50
    
    if len(food_distances) > 0:
        closest_food = min(food_distances)
        score -= closest_food * 1/40

    if not current_game_state.hasFood(new_position[0], new_position[1]):
        score -= 12.5
    else:
        score += 25
    
    
    for ghostState in new_ghost_states:
        if ghostState.scaredTimer >= 1:
            manhattan_distance = manhattanDistance(new_position, ghostState.getPosition())
            if manhattan_distance == 0:
                score += 200
            elif manhattan_distance < ghostState.scaredTimer:
                score += (1/manhattan_distance) * 200
    
    if current_game_state.isLose():
        score = -2000

    return score
    

def scoreEvaluationFunction(currentGameState: GameState):
    return heuristic(currentGameState)


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
        # chosen_action = legal_actions[max_indexes[0]]
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
        # chosen_action = legal_actions[min_indexes[0]]
        return min_score, chosen_action
    
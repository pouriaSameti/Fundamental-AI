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


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn="scoreEvaluationFunction", depth="2", time_limit="6"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.time_limit = int(time_limit)


class AIAgent(MultiAgentSearchAgent):
    
    
    def getAction(self, gameState: GameState):
        pass
        """
        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        # TODO: Your code goes here
        # util.raiseNotDefined()
    def minimax(self, game_state: GameState, depth: int, agent_index: int, pac_turn = True):
        if depth == 0 or game_state.isWin() or game_state.isLose():
            return game_state.getScore()
        
        if pac_turn:
            return self.pac_value()
        else: 
            return self.rooh_value()
    
    def pac_value(self, game_state: GameState, depth: int, agent_index: int):
        legal_actions = game_state.getLegalActions(agent_index)
        scores = []
        for action in legal_actions:
            next_game_state = game_state.generateSuccessor(agent_index, action)
            scores.append(self.minimax(next_game_state, depth, agent_index = agent_index + 1, pac_turn = False)[0])
        
        max_score = max(scores)
        max_indexs = [i for i, score in enumerate(scores) if score == max_score]
        chosen_action = action[random.choice(max_indexs)]
        return max_score, chosen_action
    
    
    
# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        # inpired by computeQValueFromValues and computeActionFromValues (written prior) below
        # apply computeQValues and within there replace call to computeQValues with the body of that function
        while self.iterations > 0:
            temps = self.values.copy()#save original values in temporary 
            states = self.mdp.getStates() #get all states
            for aState in states: #for each state use mdp to get all possible actions
                allActions = self.mdp.getPossibleActions(aState)
                possibleVals = []
                for action in allActions:#computeActionsFromValues do compute q-values from values
                    endStates = self.mdp.getTransitionStatesAndProbs(aState, action)
                    weighted = 0
                    for s in endStates: # for each end state calculate weigted average/q value
                        nextState = s[0] #get next state p 
                        prob = s[1] #get probability
                        reward = self.mdp.getReward(aState, action, nextState)
                        weighted += (prob * (reward + (self.discount * temps[nextState]))) 
                    possibleVals.append(weighted)
                if len(possibleVals) != 0:
                    self.values[aState] = max(possibleVals)
            self.iterations -= 1 #decrement until eventually iterations <= 0
            

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        endStates = self.mdp.getTransitionStatesAndProbs(state, action)
        weighted = 0
        for s in endStates:
            nextState = s[0]
            prob = s[1]
            reward = self.mdp.getReward(state, action, nextState)
            weighted += (prob* (reward + (self.discount * self.values[nextState])))

        return weighted

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state): #make sure game isn't over
            return None
        allActions = self.mdp.getPossibleActions(state)#get all actions
        endAction = ""
        maxSum = float("-inf") #placeholder val
        for action in allActions:
            weighted = self.computeQValueFromValues(state, action)#get wieghted average
            if (maxSum == float("-inf") and action == "") or weighted >= maxSum: #not yet assigned or bigger than current max
                endAction = action
                maxSum = weighted

        return endAction


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        for i in range(self.iterations):
            index = i %  len(self.mdp.getStates())
            state = self.mdp.getStates()[index]
            top = self.computeActionFromValues(state)
            if not top:
                qval = 0
            else:
                qval = self.computeQValueFromValues(state, top)
            self.values[state] = qval

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def getAllQVals(self, state):
        actions = self.mdp.getPossibleActions(state)  # All possible actions from a state
        qVals = util.Counter()  #  action: qValue pairs

        for action in actions:
            qVals[action] = self.computeQValueFromValues(state, action)
        return qVals

    def runValueIteration(self):
        pq = util.PriorityQueue()
        allStates = self.mdp.getStates()
        pred = {} #dictionary of a states predeccessors
        for state in allStates:
            pred[state]=set()
        for state in allStates:
            allActions=self.mdp.getPossibleActions(state)
            for action in allActions:
                possibleNextStates = self.mdp.getTransitionStatesAndProbs(state, action)
                for posState in possibleNextStates:
                    if posState[1]>0:
                        pred[posState[0]].add(state)
        for state in allStates: 
            allQValues = self.getAllQVals(state)
            if len(allQValues) > 0:
                maxQ = allQValues[allQValues.argMax()]
                diff = abs(self.values[state] - maxQ)
                pq.push(state, -diff)
        for i in range(self.iterations):
            if pq.isEmpty():
                return None
            state = pq.pop()
            allQValues = self.getAllQVals(state)
            maxQ = allQValues[allQValues.argMax()]
            self.values[state] = maxQ
            for p in pred[state]:

                pQValues = self.getAllQVals(p)
                maxQ = pQValues[pQValues.argMax()]
                diff = abs(self.values[p] - maxQ)
                if diff > self.theta:
                    pq.update(p, -diff)



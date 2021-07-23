import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

RUNS = 200
STEPS = 10000        
k = 10

def make_method(epsilon, stepsize=0):
    """
    if stepsize not constant, assign stepsize in "main"

    """
    averageReward = np.empty(STEPS)
    percentageOptimalAction = np.empty(STEPS)
    Q = np.zeros(k) 

    method = {
        "epsilon" : epsilon,
        "stepsize" : stepsize,
        "Q" : Q,
        "averageReward" : averageReward,
        "percentageOptimalAction" : percentageOptimalAction
    }
    return method

def choose_action(epsilon, Q):
    strategies = ["exploit", "explore"]
    probabilties = [1 - epsilon, epsilon]
    strategy = np.random.choice(a=strategies, size=None, replace=True, p=probabilties)

    if strategy == "exploit":
        return np.argmax(Q)
    else:
        actions = [i for i in range(k)]
        action = np.random.choice(a=actions)
        return action
          
def update_Q(stepsize, Q, action, reward):
    Q[action] = Q[action] + stepsize * (reward - Q[action])

def random_walks(q):
    for i in range(np.size(q)):
        q[i] += np.random.normal(loc=0.0, scale=0.01, size=None)

def average_reward(averageReward, currentReward, step, run):
    averageReward[step] = averageReward[step] + (1 / (run + 1)) * (currentReward - averageReward[step])

def percentage_optimal_action(q, action, percentageOptimalAction, step, run):
    currentPercentage = 0
    if q[action] == max(q):
        currentPercentage = 100
    percentageOptimalAction[step] = percentageOptimalAction[step] + (1 / (run + 1)) * (currentPercentage - percentageOptimalAction[step])

def plot(methods):
    x = [(i + 1) for i in range(STEPS)]

    fig, (ax1, ax2) = plt.subplots(2)

    for method in methods:
        ax1.plot(x, methods[method]["averageReward"], label=method)   
        ax2.plot(x, methods[method]["percentageOptimalAction"], label=method)

    ax1.legend()
    ax1.set(xlabel='Steps', ylabel='Average reward')

    ax2.legend()
    ax2.set(xlabel='Steps', ylabel='% Optimal action')
    
    fig.savefig(str(k) + "-armed_testbed.png")
    plt.show()
    plt.close()
    
methods = {
    "average sample e=0.1" : make_method(0.1),   
    "a=0.1 e=0.1 " : make_method(0.1, 0.1),
    "a=0.1 e=0.01" : make_method(0.01, 0.1)
}

for run in range(RUNS):
    #t1 = perf_counter() 
    q = np.zeros(k)  
    for method in methods:
        methods[method]["Q"] = np.zeros(k) 

    for step in range(STEPS):   
        random_walks(q)

        methods["average sample e=0.1"]["stepsize"] = 1 / (step + 1)   #assigning non constant stepsizes

        for method in methods:
            action = choose_action(methods[method]["epsilon"], methods[method]["Q"]) 
            reward = np.random.normal(loc=q[action], scale=1, size=None)       
            average_reward(methods[method]["averageReward"], reward, step, run)
            percentage_optimal_action(q, action, methods[method]["percentageOptimalAction"], step, run)
            update_Q(methods[method]["stepsize"], methods[method]["Q"], action, reward)

    #t2 = perf_counter()
    #print(t2 - t1)

plot(methods)
 
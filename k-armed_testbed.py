import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from abc import ABCMeta, abstractmethod
from itertools import product

RUNS = 100 #the data are averages over RUNS number of runs with different bandit problems 
STEPS = 1000 #a run lasts STEPS number of steps. At each step we choose a bandit      
K = 10  #number of bandits


def random_max_index(array):
    max_indexes = []
    for i in range(K):
        if array[i] == np.amax(array):
            max_indexes.append(i)
    return np.random.choice(a=max_indexes)


class ArmedBandits():
    '''
    The k armed bandits

    Parameters
    ----------
    initial_q : array_like (length=k)
        the initial q value for each of the k armed bandits
   
    Attributes
    ----------
    initial_q : list
        the initial q value for each of the k armed bandits
    q : list
        the q values for each of the k armed bandits 
    '''

    def __init__(self, initial_q):
        self.initial_q = initial_q
        self.q = initial_q

    def reward(self, action):
        rewards =  [np.random.normal(loc=self.q[i], scale=1, size=None) for i in range(K)]
        return rewards[action]

    def change_q(self):
        for i in range(np.size(self.q)):
            self.q[i] += np.random.normal(loc=0.0, scale=0.01, size=None) #random walks
    
    def initialize_q(self):
        self.q = self.initial_q


class Method(metaclass=ABCMeta):
    '''
    An action-value method

    Parameters
    ----------
    name : string
        the name of the method
   
    Attributes
    ----------
    name : string
        the name of the method
    average_rewrd : ndarray (shape=(STEPS))
        the average of the rewards we got at a particular step over the runs
    percentage_optimal_action : ndarray (shape=(STEPS))
        the percentage we chose the optimal action at a particular step
    '''

    def __init__(self, name):
        self.average_reward = np.zeros(STEPS)
        self.percentage_optimal_action = np.zeros(STEPS)
        self.name = name

    def update_average_reward(self, reward, step, run):
        self.average_reward[step] = self.average_reward[step] + (1 / (run+1)) * (reward - self.average_reward[step])

    def update_percentage_optimal_action(self, q, action, step, run):
        current_percentage = 0
        if q[action] == max(q):
            current_percentage = 100
        self.percentage_optimal_action[step] = self.percentage_optimal_action[step] + (1 / (run+1)) * (current_percentage - self.percentage_optimal_action[step])

    @abstractmethod
    def choose_action(self):
        pass

    @abstractmethod
    def update_estimates(self, *args, **kwargs):
        pass

    @abstractmethod
    def initialize(self):
        pass


class eGreedy(Method):
    '''
    An epsilon Greedy method (subclass of Method)

    Parameters
    ----------
    epsilon : float
        the epsilon value (ε)
    stepsize : float
        the stepsize (α)
    initial_Q : array_like
        the initial Q* values
    name : string
        the name of the method
   
    Attributes
    ----------
    epsilon : float
        the epsilon value (ε)
    stepsize : float
        the stepsize (α)
    initial_Q : array_like (length=k)
        the initial Q* values
    Q : array_like 
        the Q* values 
    '''
    def __init__(self, epsilon, stepsize=0.1, initial_Q=np.zeros(K), name="ε greedy"):       
        super().__init__(name)
        self.epsilon = epsilon
        self.stepsize = stepsize
        self.Q = initial_Q
        self.initial_Q = initial_Q

    def choose_action(self):
        strategies = ["exploit", "explore"]
        probabilities = [1 - self.epsilon, self.epsilon]
        strategy = np.random.choice(a=strategies, size=None, replace=True, p=probabilities)

        if strategy == "exploit":
            return random_max_index(self.Q)
        else:
            actions = [i for i in range(K)]
            action = np.random.choice(a=actions)
            return action

    def update_estimates(self, action, reward, *args, **kwargs):        
        self.Q[action] = self.Q[action] + self.stepsize * (reward - self.Q[action])

    def initialize(self):
        self.Q = self.initial_Q


class UCB(Method):
    '''
    An UCB method (subclass of Method)

    Parameters
    ----------
    c : float
        the c value
    stepsize : float
        the stepsize (α)
    initial_Q : array_like
        the initial Q* values
    name : string
        the name of the method
   
    Attributes
    ----------
    c : float
        the c value
    stepsize : float
        the stepsize (α)
    Q : array_like
        the Q* values 
    times_action : ndarray
        the times each action has been selected
    total_estimates : array_like
        the total estimates for each action including the Q* and the uncertainty
    initial_Q : array_like
        the initial Q* values
    '''
    def __init__(self, c, stepsize=0.1, initial_Q=np.zeros(K), name="UCB"):
        super().__init__(name)
        self.c = c
        self.stepsize = stepsize
        self.Q = initial_Q
        self.times_action = np.zeros(K)
        self.total_estimates = initial_Q
        self.initial_Q = initial_Q

    def choose_action(self):
        return random_max_index(self.total_estimates)
    
    def update_estimates(self, action, reward, step, *args, **kwargs):
        self.Q[action] = self.Q[action] + self.stepsize * (reward - self.Q[action])
        self.times_action[action] += 1
        uncertainty = np.array([100000 for i in range(K)])         # a much bigger number than the expected estimates so this action certainly gets selected
        for i in range(K):
            if self.times_action[i] > 0:
                uncertainty[i] = self.c * np.sqrt(np.log(step + 1) / self.times_action[i])
            self.total_estimates[i] = self.Q[i] + uncertainty[i] 

    def initialize(self):
        self.Q = self.initial_Q
        self.times_action = np.zeros(K)
        self.total_estimates = self.initial_Q        


class GradientBandit(Method):
    '''
    An UCB method (subclass of Method)

    Parameters
    ----------
    stepsize : float
        the stepsize (α)
    initial_H : array_like
        the initial H values
    name : string
        the name of the method
   
    Attributes
    ----------
    stepsize : float
        the stepsize (α)
    H : array_like
        the H values
    initial_Q : array_like
        the initial Q* values
    avgReward : float
        the average of the rewards we got over the steps for a particular run
    probability : ndarray
        the probability for each action to be selected
    initial_H : array_like
        the initial H values
    '''
    def __init__(self, stepsize=0.1, initial_H=np.zeros(K), name="gradient bandit"):
        super().__init__(name)
        self.stepsize = stepsize
        self.H = initial_H
        self.avgReward = 0      #####################################################
        self.probability = np.exp(self.H) / np.sum(np.exp(self.H))
        self.initial_H = initial_H

    def softmax(self):
        self.probability = np.exp(self.H) / np.sum(np.exp(self.H))
        
    def choose_action(self):
        actions = [i for i in range(K)]
        return np.random.choice(a=actions, size=None, replace=True, p=self.probability)
    
    def update_estimates(self, action, reward, step, *args, **kwargs): 
        for i in range(K):
            if i == action:
                self.H[action] = self.H[action] + self.stepsize * (reward - self.avgReward) * (1 - self.probability[action])
            else:
                self.H[i] = self.H[i] - self.stepsize * (reward - self.avgReward) * self.probability[i]
        
        self.avgReward = self.avgReward + (1 / (step+1)) * (reward - self.avgReward)
        
        self.softmax() 

    def initialize(self):
        self.H = self.initial_H
        self.avgReward = 0
        self.softmax()


class AverageStudy():
    '''
    An average performance study of the action-value methods

    Parameters
    ----------
    methods : array_like
        contains Method objects (eGreedy, UCB, GradientBandit), the methods we will study
    bandits : ArmedBandits
        the ArmedBandits object we study our methods on

    Attributes
    ----------
    methods : array_like
        contains Method objects (eGreedy, UCB, GradientBandit), the methods we will study
    bandits : ArmedBandits
        the ArmedBandits object we study our methods on  
    '''
    def __init__(self, methods, bandits):
        self.methods = methods
        self.bandits = bandits

    def study(self):
        '''
        Does the study. Computes the average reward, the percentage of optimal action chosen and the estimation of the value of the action chosen at each step.
        '''
        for run in range(RUNS):
            print("current run:" + str(run))
            #t1 = perf_counter() 
            self.bandits.initialize_q() 
            for i in range(len(self.methods)):
                self.methods[i].initialize()

            for step in range(STEPS):   
                self.bandits.change_q()

                for i in range(len(self.methods)):
                    action = self.methods[i].choose_action()
                    reward = self.bandits.reward(action)
                    self.methods[i].update_average_reward(reward, step, run)
                    self.methods[i].update_percentage_optimal_action(self.bandits.q, action, step, run)
                    self.methods[i].update_estimates(action, reward, step)
            
            #t2 = perf_counter()
            #print(t2 - t1)

    def plot(self):
        '''
        Plots the results of the study.\n
        1st plot : x axis : the steps, y axis : average reward at aech step\n
        2nd plot : x axis : the steps, y axis : percentage of optimal action chosen at each step
        '''
        x = np.arange(1, STEPS + 1)

        fig, (ax1, ax2) = plt.subplots(2)

        for i in range(len(self.methods)):
            ax1.plot(x, self.methods[i].average_reward, label=self.methods[i].name)   
            ax2.plot(x, self.methods[i].percentage_optimal_action, label=self.methods[i].name)

        ax1.legend()
        ax1.set(xlabel='Steps', ylabel='Average reward')

        ax2.legend()
        ax2.set(xlabel='Steps', ylabel='% Optimal action')
        
        fig.savefig(str(K) + "-armed_testbed.png")
        plt.show()
        plt.close()



class ParametricMethod(metaclass=ABCMeta):
    '''
    A parametric action-value method, parametric as it represents a method we want to do parametry study on

    Parameters
    ----------
    methods : array_like (3D)
        contains the Method objects (eGreedy, UCB, GradientBandit) with the desired parameter values.
        Each object has a particular setting of the parameters we treat as random variables (3 parameters hence 3D)
    name : string
        the name of the parametric method

    Attributes
    ----------
    methods : array_like
        contains the Method objects (eGreedy, UCB, GradientBandit) with the desired parameter values.
        Each object has a particular setting of the parameters we treat as random variables
    name : string
        the name of the parametric method
    '''
    def __init__(self, name):
        self.name = name
        self.methods = self.create_methods()

    @abstractmethod
    def create_methods(self):
        pass

    def parameters_index(self):
        """returns an arrow with the indexes such that parameters[index] is a parameter we want to study by"""
        indexes = []
        for i in range(len(self.parameters)):
            if len(self.parameters[i]) > 1:
                indexes.append(i)
        return indexes


class Parametric_eGreedy(ParametricMethod):
    '''
    A parametric epsilon greedy method

    Parameters
    ----------
    name : string
        the name of the parametric method
    epsilon_values : array_like
        contains the values we want the parameter epsilon to take for the parameter study.
        If we do not want to treat epsilon as a random variable still pass the single desired value for epsilon in an array_like container
    alpha_values : array_like
        contains the values we want the parameter stepsize(alpha) to take for the parameter study. 
        If we do not want to treat stepsize as a random variable still pass the single desired value for stepsize in an array_like container
    initial_Q_values : array_like
        contains the values we want the parameter initial_Q to take for the parameter study. 
        All positions in initial_Q will be filled with the same value, an initial_Q_values value.
        e.g initial_Q_values = [0, 1] then initial_Q  will take the values [0 for i in range(K)] and [1 for i in rnage(K)]
        If we do not want to treat initial_Q as a random variable still pass the single desired value for initial_Q in an array_like container

    Attributes
    ----------
    parameters : list (2D)
        contains the arrays with the values of each parameter 
    parameters_names : list
        contains the names of those parametrs in string form
    '''
    def __init__(self, name, epsilon_values, alpha_values, initialQ_values):
        self.parameters = [epsilon_values, alpha_values, initialQ_values]
        self.parameters_names = ["ε", "α", "initial Q values"]
        super().__init__(name)

    def create_methods(self):
        methods = np.empty((len(self.parameters[0]), len(self.parameters[1]), len(self.parameters[2])), dtype=object)
        for i in range(len(self.parameters[0])):
            for j in range(len(self.parameters[1])):
                for k in range(len(self.parameters[2])):
                    methods[i][j][k] = eGreedy(self.parameters[0][i], self.parameters[1][j], [self.parameters[2][k] for i in range(K)], self.name)
        return methods


class ParametricUCB(ParametricMethod):
    '''
    A parametric UCB method

    Parameters
    ----------
    name : string
        the name of the parametric method
    c_values : array_like
        contains the values we want the parameter c to take for the parameter study.
        If we do not want to treat c as a random variable still pass the single desired value for c in an array_like container
    alpha_values : array_like
        contains the values we want the parameter stepsize(alpha) to take for the parameter study. 
        If we do not want to treat stepsize as a random variable still pass the single desired value for stepsize in an array_like container
    initial_Q_values : array_like
        contains the values we want the parameter initial_Q to take for the parameter study. 
        All positions in initial_Q will be filled with the same value, an initial_Q_values value.
        e.g initial_Q_values = [0, 1] then initial_Q  will take the values [0 for i in range(K)] and [1 for i in rnage(K)]
        If we do not want to treat initial_Q as a random variable still pass the single desired value for initial_Q in an array_like container

    Attributes
    ----------
    parameters : list (2D)
        contains the arrays with the values of each parameter 
    parameters_names : list
        contains the names of those parametrs in string form
    '''
    def __init__(self, name, c_values, alpha_values, initialQ_values):
        self.parameters = [c_values, alpha_values, initialQ_values]
        self.parameters_names = ["c", "α", "initial Q values"]
        super().__init__(name)

    def create_methods(self): 
        methods = np.empty((len(self.parameters[0]), len(self.parameters[1]), len(self.parameters[2])), dtype=object)      
        for i in range(len(self.parameters[0])):
            for j in range(len(self.parameters[1])):
                for k in range(len(self.parameters[2])):
                    methods[i][j][k] = UCB(self.parameters[0][i], self.parameters[1][j], [self.parameters[2][k] for i in range(K)], self.name)
        return methods


class ParametricGradientBandit(ParametricMethod):
    '''
    A parametric gradient bandit method

    Parameters
    ----------
    name : string
        the name of the parametric method
    alpha_values : array_like
        contains the values we want the parameter stepsize(alpha) to take for the parameter study. 
        If we do not want to treat stepsize as a random variable still pass the single desired value for stepsize in an array_like container
    
    Attributes
    ----------
    parameters : list (2D)
        contains the arrays with the values of each parameter 
    parameters_names : list
        contains the names of those parametrs in string form
    '''
    def __init__(self, name, alpha_values):
        self.parameters = [alpha_values]
        self.parameters_names = ["α"]
        super().__init__(name)
    
    def create_methods(self):
        methods = np.empty((len(self.parameters[0]), 1, 1), dtype=object)
        for i in range(len(self.parameters[0])):
            methods[i][0][0] = GradientBandit(self.parameters[0][i], name=self.name)       
        return methods


class ParameterStudy():
    '''
    A parametric study of the action-value methods

    Parameters
    ----------
    dimensions : int (2 or 3)
        the dimension of the graph. For dim = 2, we get a 2D graph and we must treat one parameter as variable per method
        (for each method enter multiple values only for one parameter). For dim = 3, we get a 3D graph and we must treat exactly 2 parameters
        as variables per method (for each method enter multiple values for exactly two parameters)
    parametric_methods : array_like
        contains parametricMethod objects (Parametric_eGreedy, ParametricUCB, ParametricGradientBandit), the methods we will study
    bandits : ArmedBandits
        the ArmedBandits object we study our methods on

    Attributes
    ----------
    dim : int (2 or 3)
        the dimension of the graph. For dim = 2, we get a 2D graph and we must treat one parameter as variable per method
        (for each method enter multiple values only for one parameter). For dim = 3, we get a 3D graph and we must treat exactly 2 parameters
        as variables per method (for each method enter multiple values for exactly two parameters)
    methods : array_like
        contains Method objects (eGreedy, UCB, GradientBandit), the methods we will study
    bandits : ArmedBandits
        the ArmedBandits object we study our methods on  
    '''
    def __init__(self, dimensions, parametric_methods, bandits):
        self.dim = dimensions #dimD graph as result
        self.parametric_methods = parametric_methods
        self.bandits = bandits

    def study(self):
        '''
        Does the parameter study. Performs an average performance study for every set of parameters for all parametric methods.
        '''
        def methods_flat():
            all_methods = [self.parametric_methods[i].methods for i in range(len(self.parametric_methods))]
            methods_flat = []
            for i in range(len(all_methods)):
                arr = np.hstack(all_methods[i].flatten())
                methods_flat.append(arr)
            methods_flat = np.array(methods_flat, dtype=object)
            methods_flat = np.hstack(methods_flat.flatten())
            return methods_flat

        all_methods = methods_flat()
        avg_study = AverageStudy(all_methods, self.bandits)
        avg_study.study()

    def plot(self):   
        '''
        Plots the results of the study.
        When dim == 2 : makes a 2d plot, x axis : the set of values af the parameter-variable, y axis : average reward on STEPSth step.
        When dim == 3 : makes a 3d plot, x axis : the set of values af the 1st parameter-variable, y axis : the set of values af the 2nd parameter-variable, z axis : average reward on STEPSth step.
        '''    
        avg_reward_STEPS = [[] for i in range(len(self.parametric_methods))]
        for i in range(len(self.parametric_methods)):
            for j in range(np.shape(self.parametric_methods[i].methods)[0]):
                for k in range(np.shape(self.parametric_methods[i].methods)[1]):
                    for m in range(np.shape(self.parametric_methods[i].methods)[2]):
                        avg_reward_STEPS[i].append(self.parametric_methods[i].methods[j][k][m].average_reward[STEPS - 1])

        if self.dim == 2:
            fig, ax = plt.subplots()

            x_label=""

            for i in range(len(avg_reward_STEPS)):
                x = self.parametric_methods[i].parameters[self.parametric_methods[i].parameters_index()[0]]
                y = avg_reward_STEPS[i]

                ax.plot(x, y, label=self.parametric_methods[i].name)

                x_label += self.parametric_methods[i].parameters_names[0] + "  "

            ax.legend()
            ax.set(xlabel=x_label, ylabel="Average Reward on " + str(STEPS) + "th step")

        if self.dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            x_label=""
            y_label=""

            for i in range(len(avg_reward_STEPS)):
                x_vals = self.parametric_methods[i].parameters[self.parametric_methods[i].parameters_index()[0]]
                y_vals = self.parametric_methods[i].parameters[self.parametric_methods[i].parameters_index()[1]]
                xy = list(product(x_vals, y_vals))
                x = np.empty(len(xy))
                y = np.empty(len(xy))
                for j in range(len(xy)):
                    x[j] = xy[j][0]
                    y[j] = xy[j][1] 
                z = np.array(avg_reward_STEPS[i])

                surf = ax.plot_trisurf(x, y, z, label=self.parametric_methods[i].name) 

                x_label += self.parametric_methods[i].parameters_names[self.parametric_methods[i].parameters_index()[0]] + "  "
                y_label += self.parametric_methods[i].parameters_names[self.parametric_methods[i].parameters_index()[1]] + "  "

                surf._facecolors2d = surf._facecolor3d
                surf._edgecolors2d = surf._edgecolor3d
                           
            ax.legend()
            ax.set(xlabel=x_label, ylabel=y_label, zlabel="Average Reward on " + str(STEPS) + "th step")             

        fig.savefig("parameter_study.png")
        plt.show()
        plt.close()



if __name__ == '__main__':

    bandits = ArmedBandits([0 for i in range(K)])

    parametric_methods = [
        Parametric_eGreedy("ε-greedy", [0.1, 0.3, 0.5, 0.8], [0.1, 0.4], [0]),
        #ParametricUCB("UCB", [1, 2, 3, 4], [0.1, 0.5], [0]),
        #ParametricGradientBandit("gradient bandit", [0.1, 0.2, 0.5]),
        #Parametric_eGreedy("greedy with optimistic initialization", [0.1], [0.1], [0, 2, 4, 6])
    ]

    methods = [
        eGreedy(epsilon=0.1,  name="e-greedy, e=0.1, a=0.1"),
        eGreedy(epsilon=0, initial_Q=[5 for i in range(K)], name="greedy with optimistic initialization, a=0.1"),
        UCB(c=2, name="UCB, c=1, a=0.1"),
        GradientBandit(name="gradient bandit, a=0.1")
    ]
                
    study = ParameterStudy(3, parametric_methods, bandits)
    #study = AverageStudy(methods, bandits)
    study.study()
    study.plot()

   

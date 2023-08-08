import torch.nn as nn
from pymoo.core.algorithm import Algorithm
from pymoo.operators.sampling.rnd import FloatRandomSampling
from torch.distributions import Normal, Uniform
from pymoo.core.initialization import Initialization
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.core.population import Population
from pymoo.operators.repair.bounds_repair import is_out_of_bounds_by_problem
from pymoo.core.repair import NoRepair
from torch import optim
import torch
import numpy as np
from pymoo.util.optimum import filter_optimum

class Policy(nn.Module):
    def __init__(self,input_shape):
        super().__init__()
        print("input_shape",input_shape)
        self.model = nn.Sequential(
            nn.Linear(input_shape[0],32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,32),
        )
        self.mean = nn.Sequential(nn.Linear(32, input_shape[0]),
                                    nn.Tanh())                    # tanh squashed output to the range of -1..1
        self.variance =nn.Sequential(nn.Linear(32, input_shape[0]),
                                        nn.Softplus())
    def forward(self,x):
        x = self.model(x)
        return self.mean(x), self.variance(x)

class MonteCarloGradientPolicyAlgorithm(Algorithm):
    def __init__(self,
                 gamma=0.05,
                 alpha=0.1,
                 num_rounds=20,
                 sampling=FloatRandomSampling(),
                 repair=NoRepair(),
                 step_ratio = 0.05,
                 **kwargs):
        
        """
        Parameters
        ----------
        env : 
            The environment to be used in the algorithm.
        policy : {Policy}
            The policy to be used in the algorithm.
        gamma : float, optional
            The discount factor used in the algorithm. The default is 0.99.
        alpha : float, optional
            The learning rate used in the algorithm. The default is 0.01.
        num_episodes : int, optional
            The number of episodes to be run in the algorithm. The default is 100.
        sample_size : int, optional
            The number of samples to be generated from the problems and used in the acquisition function. 
            The default is 10.
        sampling : {Sampling}, optional
            The sampling method used to generate the initial samples. The default is FloatRandomSampling().
        """
         
        super().__init__(**kwargs)

        self.gamma = gamma
        self.alpha = alpha
        self.num_rounds = num_rounds
        self.sampling = sampling
        self.repair = repair
        self.step_ratio = step_ratio
        self.initialization = Initialization(sampling)
        self.survival = RankAndCrowdingSurvival()
        self.crossover = SimulatedBinaryCrossover(n_offsprings=1)
        self.is_constraint_model = False
        self.optimizer = None
        self.model = None
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.steps_taken = []
        

    def _setup(self, problem, **kwargs):
        self.model = Policy(np.array([self.problem.n_var]))
        self.optimizer = optim.Adam(self.model.parameters(), lr = 5e-3)
        self.step_size = (self.problem.xu - self.problem.xl)*self.step_ratio
        
    def _initialize_infill(self):
        return self.initialization.do(self.problem, 1, algorithm=self)

    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills=infills, **kwargs)

    def _infill(self):
        state = self.get_starting_point()
        normalized_state = self.custom_state_normalization(state)
        termination_rounds = self.num_rounds 
        steps = 0
        ep_rewards = 0
        batch_rewards = []
        log_probs = []

        while steps < termination_rounds:
            #print("self.opt.get(F)",self.opt.get("F"))
           # print("self.problem.evaluate(np.array(state))[0]",self.problem.evaluate(np.array(state))[0])
            a, log_p = self.action(torch.Tensor(normalized_state).unsqueeze(0))
            cliped_a = np.clip(a, -1, 1)
            action_vector = self.step_size* (cliped_a + np.array([1])) - self.step_size
            log_probs.append(log_p)
            new_state, reward,_ = self.step(action_vector,state)
            normalized_new_state = self.custom_state_normalization(new_state)
            batch_rewards.append(reward)
            state = new_state
            normalized_state = normalized_new_state
            ep_rewards += reward
            steps +=1
            state = new_state
        self.rewards.append(ep_rewards)
        self.steps_taken.append(steps)
        print("Episode: {} --- Rewards: {} --- Steps: {}".format(self.n_iter, ep_rewards, steps))
        self.update_policy(self.n_iter, self.optimizer, batch_rewards, log_probs)

        
        return self.pop
        
    def _advance(self, infills=None, **kwargs):
        return super()._advance(infills=infills, **kwargs)
    
    
    def _finalize(self):
        return super()._finalize()
    
    def action(self, state):
        # simple pytorch aproach for action-selection and log-prob calc 
        #action_parameters = model(s)
        
        mu,variance = self.model(state)
        mu = torch.nan_to_num(mu[0])
        variance = torch.nan_to_num(variance[0])
        sigma = torch.sqrt(variance)
        
        #print("sigma:", sigma)
        m = Normal(mu, sigma)

        a = m.sample()
 
        log_p = m.log_prob(a)
  
        return a.tolist(), log_p
    
    def add_individual(self, individual):
        self.pop = Population.new(X=np.vstack((self.pop.get("X"), individual)))
        #print("pop",self.pop)
    def get_rewards(self, current_state, new_state):
        
        #print("current_state",current_state)
        #print("new_state",new_state)
        is_infesible = False
        current_state_value = self.problem.evaluate(current_state, return_values_of=["F"])
        new_state_value = self.problem.evaluate(new_state, return_values_of=["F"])
        diff = new_state_value - current_state_value
        eucli_dist = self.euclidean_distance(current_state_value,new_state_value)
        c1, c2, c3, c4, c5, c6 = 3, 1, -1, -3, -3, -1

        domain_penalty = self.domain_penalty_function(new_state)
        constraint_penalty = self.constraint_penalty_function(new_state)
        if domain_penalty > 0 or constraint_penalty > 0:
  
            is_infesible = True

            return c5*eucli_dist + c6*(domain_penalty + constraint_penalty), is_infesible
        elif np.all(diff < 0):

            self.add_individual(new_state)
            return c1*eucli_dist, is_infesible
        elif np.all(diff > 0):

            return c4*eucli_dist, is_infesible
        elif np.all(diff == 0):

            return c3, is_infesible
        else:
            return c2*eucli_dist, is_infesible
    
    def domain_penalty_function(self, state):
        return  np.sum((np.maximum(0, state - self.problem.xu)**2))**0.5 + np.sum((np.maximum(0, self.problem.xl - state)**2)**0.5)
    
    def constraint_penalty_function(self, state):
        return np.sum((np.maximum(0, self.problem.evaluate(state, return_values_of=["G"]))**2))**0.5 + np.sum((self.problem.evaluate(state, return_values_of=["H"])**2)**0.5)
    
    def get_starting_point(self):
        if self.pop.size <= 2 or np.random.random_sample() > 0.5:
            return self.initialization.do(self.problem, 1, algorithm=self).get("X")[0]
            
            #print("point",point, self.constraint_penalty_function(point))
            #print("n_point",point, self.custom_state_normalization(point))
            #return point
            #return self.initialization.do(self.problem, 1, algorithm=self).get("X")[0]
        else:
            new_parents = self.survival.do(self.problem, self.pop, n_survive=2)
            new_state = self.crossover.do(self.problem, [new_parents]).get("X")[0]
            return new_state
    
    def custom_state_normalization(self, state):
        max = self.problem.xu
        min = (self.problem.xl + self.problem.xu)/2
        return (state-min)/(max-min)
    
    def euclidean_distance(self, current_state_value, new_state_value):
        return np.sum((new_state_value - current_state_value)**2)**0.5



    def step(self, action, state):
        current_X = state
        X_new =  current_X + action
        #print("current_X:", current_X)
        #print("action:", action)
        #print("X_new:", X_new)
        reward, is_infesible = self.get_rewards(current_X, X_new)
        
        if is_infesible:
            new_state = self.get_starting_point()
            return current_X , reward, is_infesible
        
        new_state = np.array(X_new)
        return new_state, reward, is_infesible
    
    def update_policy(self, ep, optimizer,batch_rewards,log_probs):
        R = 0
        policy_loss = []
        rewards = []
        #calc discounted Rewards
        for r in batch_rewards[::-1]: # reverses the list of rewards 
            R = r + self.gamma * R
            rewards.insert(0, R) # inserts the current reward to first position
        
        
        rewards = torch.tensor(rewards)
        # standardization to get data of zero mean and variance 1, stabilizes learning 
        rewards = (rewards - rewards.mean()) / (rewards.std() + ep)
        for log_prob, reward in zip(log_probs, rewards):
            policy_loss.append(self.alpha*(-log_prob * reward)) #baseline+
        
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),1)
        optimizer.step()

# POLICY ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the policy iteration algorithm.

## PROBLEM STATEMENT
The aim of this experiment is to find optimal policy for the mdp using policy iteration. Policy iteration includes policy evaluation and policy improvement where evaluation function is used to find optimal value function of each state and then improvement function is used to find best policy by comparing all the action value function as well as policy.

## POLICY ITERATION ALGORITHM

1) we are going to do policy evaluation of each state to get the state value function where the initial policy is defined randomly to the mdp.
2) Once we obtain convergence in the policy evaluation then implement policy improvement where we are going to find best optimal policy until the previous and current policy are same.
## POLICY IMPROVEMENT FUNCTION
### Name: Himavath M
### Register Number:212223240053
```
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob,next_state,reward,done in P[s][a]:
          Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
          new_pi=lambda s:{s:a for s, a in enumerate(np.argmax(Q,axis=1))}[s]
    return new_pi
```
## POLICY ITERATION FUNCTION

### Name:Himavath M
### Register Number:212223240053
```
def policy_iteration(P, gamma=1.0, theta=1e-10):
   random_actions=np.random.choice(tuple(P[0].keys()),len(P))
   pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]
   while True:
    old_pi = {s:pi(s) for s in range(len(P))}
    V = policy_evaluation(pi, P,gamma,theta)
    pi = policy_improvement(V,P,gamma)
    if old_pi == {s:pi(s) for s in range(len(P))}:
      break
   return V, pi
```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
</br>![image](https://github.com/user-attachments/assets/5d705298-2144-496b-9034-08f986076656)
![image](https://github.com/user-attachments/assets/13155a2f-721b-42a7-84b9-e59118560c88)
![image](https://github.com/user-attachments/assets/5773d3e7-a9c8-4a96-92c1-84c7c5ea7781)
</br>

### 2. Policy, Value function and success rate for the Improved Policy
</br>![image](https://github.com/user-attachments/assets/e7bdf56f-71d5-4f62-85e3-6f5301479524)
![image](https://github.com/user-attachments/assets/2a383496-96bb-43d5-9a69-05921ef9ca55)
![image](https://github.com/user-attachments/assets/f38762f7-b764-4d5f-8dd7-6f50c37e2e16)
</br>

### 3. Policy, Value function and success rate after policy iteration
</br>![image](https://github.com/user-attachments/assets/8f408c12-5193-4a3e-a374-967a38ccc1a3)
![image](https://github.com/user-attachments/assets/e7bf8ee8-e8a5-4258-b0a0-351b76adf41f)
![image](https://github.com/user-attachments/assets/4aabe73b-7a8f-4975-bc33-fedbc7f4d41d)
</br>
## RESULT:
 The Python program to find the optimal policy for the given MDP using the policy iteration algorithm is successfully executed.

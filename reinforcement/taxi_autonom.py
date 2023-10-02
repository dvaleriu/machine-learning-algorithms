import gym
import numpy as np
import random


env = gym.make("Taxi-v3").env
#env.reset()


state = env.encode(0,1,2,3)
#index rand,coloana,pasager(4 posibilitati),destinatie(4 pos)
print("Starea", state)
env.s = state
env.render()
print(env.P[31]) #tabelul rewardului
q_table = np.zeros([env.observation_space.n, env.action_space.n])
print(f"Shape-ul tabelului Q: {q_table.shape}")

#constante
nr_episoade = 100001

#hiperparametri
alpha = 0.1
gamma = 0.6
epsilon = 0.1

for i in range(1,nr_episoade):
    state = env.reset()
    epochs, penalties, reward = 0,0,0
    done = False
    #exlorare sau cautare
    while not done:
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample() #alegere actiune random EXPLORARE
        else:
            action = np.argmax(q_table[state]) #exploatare-cea mai amre valoare din qtable

    next_state, reard, done, info = env.step(action) #se asteapta raspunsul dupa executare actiune
    
    #valori noi qtable:
    old_value = q_table[state, action]
    next_max = np.max(q_table[next_state])
    new_value = (1-alpha) * old_value + alpha *(reward + gamma*next_max)
    q_table[state,action] = new_value

    if reward == -10:
        penalties += 1
    
    state = next_state
    epochs += 1

print("done antrenare")

#testare

total_epochs, total_penalties = 0,0
episoade = 100

for _ in range(episoade):
    state = env.reset()
    epochs, penalties, reward = 0,0,0
    done = False
    while not done:
        action = np.argmax(q_table[state])
        state,reward,done,info = env.step(action)
        if reard == -10:
            penalties += 1
        epochs += 1
        
        total_penalties += penalties
        total_epochs += epochs

print(f"Performanta agentului dupa {episoade} de episoade:")
print(f"Numarul de cazuri cand agentul nu a ridicat sau lasat corect pasagerul: [{penalties}/{episoade}]")
print(f"Numarul mediu de actiuni pe care agentul le face de la inceput pana cand ajunge cu pasagerul la destinatie: {total_epochs / episoade}")

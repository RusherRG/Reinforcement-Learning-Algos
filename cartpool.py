import gym
import random
import numpy as np
import tensorflow as tf
from statistics import mean, median
from collections import Counter

gym.envs.register(
    id='CartPoleNoLimit-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=10000,
)
env = gym.make('CartPoleNoLimit-v0')
env.reset()
goal_steps = 10000 #Score is goal_steps stop after this
score_req = 100 #Minimum score in dataset
initial_games = 2000 #Total number of iterations
drop = 0.2
epochs = 5

def init_population(initial_games,name):
    training_data = [] #stores the observations and action taken
    scores = [] #all the scores
    accepted_scores = [] #scores above score_req
    while len(accepted_scores)<initial_games:
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            #env.render()
            if len(prev_observation)>0:
                game_memory.append([prev_observation, action]) 
            #prev_observation is used because it makes sense
            #Action depends on prev_observation and not the current
            prev_observation = observation
            score += int(reward)
            if done:
                break
        scores.append(score)
        if score >= score_req:
            accepted_scores.append(score)
            print(len(accepted_scores),end='\r')
            for data in game_memory:
                if data[1] == 1:
                    output = np.array([0,1]) #right
                else:
                    output = np.array([1,0]) #left
                training_data.append([data[0], output])
        env.reset()
    
    training_data_np = np.array(training_data)
    np.save(name, training_data_np)

    print("Average score :",mean(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

#init_population()

def neural_network_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dropout(drop),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(drop),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(drop),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(drop),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(drop),
        tf.keras.layers.Dense(2, activation='softmax')
        ])
    return model

def play(model):
    number_games = 10
    scores = []
    for game in range(number_games):
        action = env.action_space.sample()
        score = 0
        choices = {0: 0, 1:0}
        env.reset()
        for _ in range(goal_steps):
            env.render()
            observation, reward, done, info = env.step(action)
            observation = np.array(observation).reshape(4,1).T
            output = model.predict(x=observation)
            # print(output)
            action = np.argmax(output)
            choices[action] += 1
            score += reward
            print("Score Game {0} = {1}".format(game+1,score),end='\r')
            if done:
                print("Score Game {0} = {1}".format(game+1,score))
                break
        scores.append(score)
        # print("Left : {0}\nRight: {1}".format(choices[0],choices[1]))
    return max(scores)

def generate_data_sets():
    print("Generating Training Data")
    training_data = init_population(initial_games, './datasets/'+str(initial_games)+'_train.npy')
    print("Training Set Size {}".format(len(training_data)))

    print("Generating Test Data")
    test_data = init_population(initial_games//100, './datasets/'+str(initial_games//100)+'_test.npy')
    print("Test Set Size {}".format(len(test_data)))
    return 

def load_datasets():
    training_data = np.load('./datasets/'+str(initial_games)+'_train.npy')
    x_train = np.array([i[0] for i in training_data])
    y_train = np.array([i[1] for i in training_data])

    test_data = np.load('./datasets/'+str(initial_games//100)+'_test.npy')
    x_test = np.array([i[0] for i in test_data]) 
    y_test = np.array([i[1] for i in test_data])
    return x_train, y_train, x_test, y_test

def train_model(x_train, y_train, x_test, y_test):
    model = neural_network_model()
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, epochs=epochs, verbose=1)
    metrics = model.evaluate(x_test,y_test) 
    print("\nTest Accuracy : {}%\n".format(round(metrics[1]*100,2)))
    #print(model.summary())
    model.save_weights("./models/epochs_{0}_accuracy_{1}.model".format(epochs, int(metrics[1]*100)))
    return model

def loadmodel(name):
    model = tf.keras.models.load_model(name)
    return model

def lets_play():
    #generate_data_sets()
    x_train, y_train, x_test, y_test = load_datasets()    
    model = train_model(x_train, y_train, x_test, y_test)
    # model = loadmodel("./epochs_10_accuracy_61.model")
    best_score = play(model)
    print("Top Score : {}".format(best_score))
    return

lets_play()





def some_random_games_first():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()

def env_test():
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()
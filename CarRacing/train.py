import gym
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from statistics import mean

gym.envs.register(
    id='CarRacingNoLimit-v0',
    entry_point='gym.envs.box2d:CarRacing',
    max_episode_steps=10000,
)


class Training:
    def __init__(self):
        self.env = gym.make('CarRacingNoLimit-v0')
        self.training_data_len = 10000
        self.num_steps = 300
        self.epochs = 3

    def random_plays(self):
        data = []
        scores = []
        while len(data) < self.training_data_len:
            self.env.reset()
            score = 0
            game_data = []
            prev_observation = None
            for _ in range(self.num_steps):
                action = self.env.action_space.sample()
                action[2] = min(action[2], 0.1)
                observation, reward, done, info = self.env.step(action)
                # self.env.render()
                if prev_observation is not None:
                    game_data.append([prev_observation, action])
                prev_observation = observation
                score += reward
                if done:
                    break
            # print("Score:", score)
            if score > 75:
                scores.append(score)
                print("# Accepted Episodes:", len(scores))
                data.extend(game_data)
                print("Dataset size:", len(data))

        print("Average score :", mean(scores))
        np.save("./datasets/{}_{}_data".format(self.training_data_len,
                                               self.num_steps), np.array(data))

    def load_data(self):
        data = np.load(
            "./datasets/{}_{}_data.npy".format(
                self.training_data_len, self.num_steps),
            allow_pickle=True)
        x, y = [], []
        for i in range(self.training_data_len):
            if type(data[i][0])!=list:
                x.append(np.asarray(data[i][0])/255.0)
                y.append(np.asarray(data[i][1]))
        x, y = np.asarray(x), np.asarray(y)
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            train_size=0.8,
                                                            test_size=0.2,
                                                            random_state=123)
        return x_train, x_test, y_train, y_test

    def generate_model(self):
        dropout = 0.1
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='linear'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(256, activation='linear'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(128, activation='linear'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(64, activation='linear'),
            tf.keras.layers.Dense(3, activation='linear')
        ])
        return model

    def train(self):
        x_train, x_test, y_train, y_test = self.load_data()
        model = self.generate_model()
        print(model.summary())
        model.compile(optimizer='SGD',
                      loss='MSLE', metrics=['mean_squared_error'])
        model.fit(x=x_train, y=y_train, epochs=self.epochs, batch_size=32, verbose=1)
        metrics = model.evaluate(x_test, y_test)
        # y_pred = model.predict(x_test)
        # print(y_pred)
        print(metrics)
        # print("\nTest Accuracy : {}%\n".format(round(metrics[1]*100, 2)))
        # print(model.summary())
        model.save_weights(
            "./models/epochs_{0}_accuracy_{1}.model".format(self.epochs, int(metrics[1]*100)))
        return model

    def load_model(self, model_name):
        model = tf.keras.models.load_model(model_name)
        return model

    def play(self, model):
        number_games = 10
        scores = []
        for game in range(number_games):
            self.env.reset()
            action = self.env.action_space.sample()
            score = 0
            for _ in range(self.num_steps):
                self.env.render()
                observation, reward, done, info = self.env.step(action)
                observation = np.array(observation).reshape(-1, 96, 96, 3)/255.0
                # print(observation.shape)
                action = model.predict(x=observation)[0]
                print(action)
                score += reward
                print("Score Game {0} = {1}".format(game+1, score), end='\r')
                if done:
                    print("Score Game {0} = {1}".format(game+1, score))
                    break
            scores.append(score)
            # print("Left : {0}\nRight: {1}".format(choices[0],choices[1]))
        return max(scores)

    def run(self):
        self.random_plays()
        model = self.train()
        # model = self.load_model("./models/epochs_5_accuracy_24.model")
        self.play(model)


trainer = Training()
trainer.run()

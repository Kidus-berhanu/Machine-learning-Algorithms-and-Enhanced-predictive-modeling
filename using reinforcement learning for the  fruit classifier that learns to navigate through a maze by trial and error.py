#kidus_berhanu
import numpy as np

# Fruit class
class Fruit:
    def __init__(self, name, reward, maze_location):
        self.name = name
        self.reward = reward
        self.maze_location = maze_location

# Maze class
class Maze:
    def __init__(self, fruit_list, maze_size):
        self.fruit_list = fruit_list
        self.maze_size = maze_size
        self.reset()
        
    def reset(self):
        self.current_location = [0, 0]
        self.maze = np.zeros(self.maze_size)
        for fruit in self.fruit_list:
            self.maze[fruit.maze_location[0], fruit.maze_location[1]] = fruit.reward
            
    def move(self, direction):
        if direction == 'up':
            self.current_location[0] -= 1
        elif direction == 'down':
            self.current_location[0] += 1
        elif direction == 'left':
            self.current_location[1] -= 1
        elif direction == 'right':
            self.current_location[1] += 1
        else:
            raise ValueError("Invalid direction")
            
    def get_reward(self):
        return self.maze[self.current_location[0], self.current_location[1]]
    
    def is_done(self):
        return self.current_location[0] < 0 or self.current_location[0] >= self.maze_size[0] or self.current_location[1] < 0 or self.current_location[1] >= self.maze_size[1]
    
# Q-Learning class
class QLearning:
    def __init__(self, maze, alpha, gamma, epsilon):
        self.maze = maze
        self.q_table = np.zeros((maze.maze_size[0], maze.maze_size[1], 4))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
    def choose_action(self):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, 4)
        else:
            return np.argmax(self.q_table[self.maze.current_location[0], self.maze.current_location[1]])
        
    def learn(self, current_reward, next_reward):
        current_q = self.q_table[self.maze.current_location[0], self.maze.current_location[1], self.last_action]
        next_q = np.max(self.q_table[self.maze.current_location[0], self.maze.current_location[1]])
        self.q_table[self.maze.current_location[0], self.maze.current_location[1], self.last_action
= current_reward + self.gamma * next_q

# Training class
class Trainer:
    def __init__(self, maze, q_learning, num_episodes):
        self.maze = maze
        self.q_learning = q_learning
        self.num_episodes = num_episodes
        
    def train(self):
        for episode in range(self.num_episodes):
            self.maze.reset()
            total_reward = 0
            while not self.maze.is_done():
                action = self.q_learning.choose_action()
                if action == 0:
                    self.maze.move('up')
                elif action == 1:
                    self.maze.move('down')
                elif action == 2:
                    self.maze.move('left')
                elif action == 3:
                    self.maze.move('right')
                reward = self.maze.get_reward()
                total_reward += reward
                self.q_learning.last_action = action
                self.q_learning.learn(reward, self.maze.get_reward())
            print("Episode {}: total reward = {}".format(episode+1, total_reward))

# Create fruit list
fruits = [Fruit("apple", 10, [2, 3]), Fruit("banana", 5, [1, 1]), Fruit("orange", 15, [0, 2])]
#this is just a sample we will use the full data set later

# Create maze
maze = Maze(fruits, [3, 4])

# Create Q-Learning agent
q_learning = QLearning(maze, alpha=0.5, gamma=0.9, epsilon=0.1)

# Create trainer
trainer = Trainer(maze, q_learning, num_episodes=100)

# Start training
trainer.train()

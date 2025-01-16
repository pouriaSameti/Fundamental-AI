
# The Artificial Intelligence Fundamentals Course Projects 

These projects are part of the **Artificial Intelligence Fundamentals course** at the **University of Isfahan**. The projects cover topics in **Machine Learning, Reinforcement Learning, and First Order Logic**, all developed under the supervision of **Dr. Hossein Karshenas**.
### Developers
- [Pouria Sameti](https://github.com/pouriaSameti)
- [Kimya Mirmoghtadaie](https://github.com/Kimya-M)

## Dijkstra & Astar

This project demonstrates the implementation of a graph data structure to apply Dijkstra's and A* (Astar) algorithms for finding the shortest path between nodes in a graph. In this implementation, the graph nodes represent airports, and the goal is to determine the shortest path between airports with the minimum cost. 

### Installation

To run this project, install the required dependencies by executing the following commands:

```python
  pip install numpy
```
```python
  pip install pandas
```

### Features
1. **Algorithm Execution Results:**
The results of the Dijkstra and A* algorithm executions are saved as .txt files in the project's folder. These results showcase the shortest path and associated costs for any given origin and destination.

2. **Algorithm Execution Results:**
On each program run, you provide an origin and destination as input. The program computes the shortest path and saves the results in two .txt files in the project's folder.

3. **Default Example:**
The current output in the repository is based on the following:
* **Origin:** Imam Khomeini International Airport
* **Destination:** Raleigh-Durham International Airport

## Linear Regression
This project demonstrates the implementation of Linear Regression **from scratch** using the NumPy and Pandas libraries.
The task involves predicting flight prices based on features such as duration, arrival time, departure time, and more.

### Pipeline
1. **Read the Dataset:** Load and preprocess the data.<br>
2. **Handle Missing Values:** Identify and address any NaN values in the dataset.<br>
3. **Process Text Features:** Transform text-based features into a usable format.<br>
4. **Feature Scaling:** Apply Min-Max Scaling (Normalization) to standardize feature values.<br>
5. **Implement Linear Regression:** Train the model using a custom implementation.<br>
6. **Evaluate the Model:** Use metrics like R²-score, Mean Squared Error (MSE), and Mean Absolute Error (MAE) to assess performance.
### Installation
To run this project, install the required dependencies using the following commands:
```python
  pip install numpy
```
```python
  pip install pandas 
```
```python
  pip install scikit-learn
```
```python
  pip install seaborn
```
<br>  

## Markov decision process(MDP)
This project implements an intelligent agent operating in a non-deterministic environment. The primary goal is to determine the optimal policy for the agent in this environment.

### Key Insight:
1. The agent has access to the Transition Model of the environment. <br>  
2. The Value Iteration algorithm is utilized to compute the optimal policy for the agent.

### Visualization
The agent operates in the **"Cliff Walking"** environment from the **gymnasium** library.<br>
Below is an example of the environment:<br>
![Cliff Walking Environment](https://github.com/pouriaSameti/Fundamental-AI/assets/91469214/80a5a081-f9dd-4c0a-9651-2ce2e669a9c0)


### Features

1. **Implementation of the Value Iteration Algorithm:**
2. **Algorithm Convergence Check:** Ensures the Value Iteration process reaches stability.
3. **Calculation of Key Metrics:** 
* Q*: Optimal state-action values.
* V*: Optimal state values.
4. **Heatmap Visualization:** Displays the scores of all states in a visually interpretable format.
5. **Convergence Plot:** Visualizes the convergence process of the algorithm over iterations.


### Installation
To run this project, install the required dependencies using the following commands:
```python
  pip install 'gymnasium[all]'
```
```python
  pip install numpy
```
```python
  pip install seaborn
```
<br>  

## Reinforcement Learning
This project demonstrates the implementation of an intelligent agent that operates in an **unknown environment**. The goal is to determine the optimal policy for the agent to maximize its performance in this environment
<br>

### Visualization
Below is an example of the **gym-maze** environment:<br>
![gym-maze Environment](https://github.com/pouriaSameti/Fundamental-AI/assets/91469214/131fd502-2c9f-41b3-9053-e088f87d1f89)


### Features

1. **Implementation of Q-Learning Algorithm:** A classic reinforcement learning approach for policy determination.
2. **Deep Q-Learning (DQN)**: Utilizes Keras for building and training deep neural networks.
3. **Action Selection Strategies:** Includes methods like:
* Epsilon-Greedy for balancing exploration and exploitation.
* Approximation Utility to suggest optimal actions based on learned policies.


### Installation
To run this project, install the required dependencies using the following commands:
```python
  pip install tensorflow
```
```python
  pip install --upgrade keras
```
```python
  pip install seaborn
```
```python
  python setup.py install
```

<br>  

## Game
This project focuses on creating an intelligent agent to play the **Pacman game**. The goal is to maximize the score by eating dots and strategically avoiding ghosts.

### Environment
* The project uses the Pacman environment developed by **UC Berkeley**.
* In this environment, the agent navigates a maze, eating small dots and large dots to score points.
* The primary challenge is avoiding ghosts, but by eating large dots, the agent temporarily gains the ability to eat ghosts and earn additional points.

![pacman Environment](https://github.com/pouriaSameti/Fundamental-AI/assets/91469214/130f90b6-9781-43ef-9ee9-0fc164773e50)

### Features
1. **Minimax Algorithm:** Implemented to solve the multi-agent decision-making problem in the game.
2. **Heuristic Method:** Developed to guide the agent towards achieving the maximum possible score and ultimately winning the game.

### Key Insight:
1. The game is a multi-agent environment, requiring strategic planning to win with the highest score.
2. Our heuristic method was rigorously tested, achieving **200 wins out of 200 games** with a single directional ghost.



### Running The Game
To run the game with intelligent ghosts, execute the following commands from the game's folder:

1. Run with 1 directional ghost for 100 games:
```python
  python pacman.py -p AIAgent -k 1 -a depth=4 -g DirectionalGhost -n 100
```
<br>

2. Run with 1 ghost with stochastic actions:
```python
  python pacman.py -p AIAgent -k 1 -a depth=4 -n 100
```

### implementation
To view the implementation of the Minimax algorithm and the heuristic method, navigate to: **Multi agent search game/multiAgents.py**<br>
<br>  

## First order logic(FOL)
This project focuses on creating a **tourist tour recommendation system** using **First Order Logic (FOL)**. The system processes user-input text about their preferences and recommends a suitable tour. The recommendation is based on a **Knowledge Base** implemented with the **Prolog** language.

### Features
1. **Graph Representation:** A graph is constructed from the input dataset using Prolog to represent tour connections.
2. **Neighbor Discovery:** Identify the first and second-degree neighbors of any node in the graph using Prolog queries.
3. **Tour Recommendation:**
* Process user input text about their preferences.
* Suggest a tour that matches their preferences based on the graph’s connected nodes and the knowledge base.
4. **Knowledge Base Design:** Use Prolog rules and facts to store relationships and enable intelligent query processing.

### Installation
To run this project, install the required dependencies using the following commands:

```python
  pip install numpy
```
```python
  pip install pandas
```
```python
  pip install pyswip
```


# The Artificial Intelligence Fundamentals Course Projects 

These projects are related to Machine Learning, Reinforcement Learning & First order logic under the supervision of Dr Hossein Karshenas at the University of Isfahan.

### Developers
- [Pouria Sameti](https://github.com/pouriaSameti)
- [Kimya Mirmoghtadaie](https://github.com/Kimya-M)

<br>  
<br> 


## Dijkstra & Astar

We implement a graph data structure to apply Dijkstra and Astar Algorithms on the nodes to find the shortest path between points in the graph.

Our nodes are airports. We want to find the shortest path between them with the minimum time cost. 

### Installation

To Run this project, run these commands:

```python
  pip install numpy
```
```python
  pip install pandas
```

### Features
1. You can see the result of the Execution of Dijkstra and Astar Algorithms in the **.txt** files in the folder of the project for any origin & destination in the Graph.
2. Every time you run this program, you should give your **origin and destination as input** to this program. Then your result was saved in **two .txt** files in the folder of the program. The current output was saved for the "Imam Khomeini International Airport" as the origin & "Raleigh Durham International Airport" as the destination.

<br>
<br>

## Linear Regression
We implement Linear Regression from scratch with Numpy and Pandas Libraries.  
Our task is the prediction of flight price concerning features like duration, arrival time, departure time, and...  

### Pipeline
1. Reading Dataset
2. Checking NAN values
3. Handling Text Features
4. Feature scaling with Min-Max scaler(Normalization)
5. Applying Linear Regression
6. Evaluation With the R2-score, MSE and MAE

### Installation

To Run this project, run these commands:

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
<br>  


## Markov decision process(MDP)
This project aims to implement an intelligent agent that exists in a non-deterministic environment. 

Our goal is to find the policy for this agent in this non-deterministic environment.

<br> 

> [!TIP]
> This agent has the **Transition Model** of the Environment. <br>  
> We use the **Value Iteration** algorithm to find the policy for the agent.

<br>

> [!IMPORTANT]
> Agent exists in the "Cliff Walking" environment from the **gymnasium**. You can see the features of this environment in the below link: https://gymnasium.farama.org/environments/toy_text/cliff_walking/


![Cliff Walking Environment](https://github.com/pouriaSameti/Fundamental-AI/assets/91469214/80a5a081-f9dd-4c0a-9651-2ce2e669a9c0)


### Features

1. implementation of **Value Iteration** algorithm
2. Checking the Convergence of the algorithm
3. Calculation of **q-star** matrix & **v-star** list
4. showing the score of every state with a heatmap
5. showing the convergence plot

### Installation

To Run this project, run these commands:

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
<br>


## Reinforcement Learning
The goal of this project is to implement an intelligent agent that pays activity in an **unknown environment**.  
Our goal is to find the policy for this agent in this unknown environment.
 
> [!TIP]
> We use the **q-learning** algorithm to find the policy for the agent.  
> We used another solution to find the policy. We use **Deep Q-Learning(DQN)** to suggest the best action in every state to the agent.

<br>

> [!IMPORTANT]
> Agent exists in the "Maze" environment from the **gym-maze**. You can see the features of this environment in the below link: https://github.com/MattChanTK/gym-maze


![gym-maze Environment](https://github.com/pouriaSameti/Fundamental-AI/assets/91469214/131fd502-2c9f-41b3-9053-e088f87d1f89)


### Features

1. implementation of **q-learning** algorithm
2. implementation of **Deep Q-Learning** with Keras.
3. selecting action with **epsilon-greedy** & **approximation_utility**

### Installation

To Run this project, run these commands:

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
<br>

## Game
The goal of this project is to implement an intelligent agent that works in the **Pacman game** environment and earns the most points.

### Environment
- In this phase, we use the Pacman environment developed by **UC Berkeley**. 
- In this environment, your agent pays activity in a maze and eats the small points and some large points. 
- In this game, the goal is to eat the dots while avoiding the ghosts. By eating the big dots, the situation changes in your favor: for a short time, you can also eat and gain ghosts.

![pacman Environment](https://github.com/pouriaSameti/Fundamental-AI/assets/91469214/130f90b6-9781-43ef-9ee9-0fc164773e50)

<br>

### Features
1. We implement the **Minimax** algorithm to solve the problem.
2. We use a Heuristic method to guide our agent to catch to the maximum score and finally win this game. 
 
<br>

> [!TIP]
>  We have a multi-agent environment. So our problem is winning the game with the maximum score.  
> We evaluated our heuristic method and caught **100 wins from 100 games** with 1 directional ghost in this game with this heuristic method.



### Running The Game
To run the game with intelligent ghosts, run this command in the folder of the game:

1. Running with 1 directional Ghost for 100 times:
```python
  python pacman.py -p AIAgent -k 1 -a depth=4 -g DirectionalGhost -n 100
```
<br>

2. Running 1 ghost with stochastic actions:
```python
  python pacman.py -p AIAgent -k 1 -a depth=4 -n 100
```

### implementation
To see the  Minimax algorithm implementation and our heuristic method, go to this path:  **Multi agent search game/multiAgents.py**

<br>  
<br>


## First order logic(FOL)
In this project, the goal is to implement a tourist tour recommendation system based on the text received from the user.  
The goal is to implement this system by using First Order Logic.

Our Task is to design a **Knowledge base** with **Prolog** Language.

### Features
1. We implement a graph With input dataset and Prolog.
2. We can find the first & Second Connected neighbors with Prolog.
3. We can receive a text from a user about his thoughts about a tour and then suggest to him a tour that matches his text.
4. This tour is based on the text input from a user and real connected nodes in the Graph.


### Installation

To Run this project, run these commands:

```python
  pip install numpy
```
```python
  pip install pandas
```
```python
  pip install pyswip
```



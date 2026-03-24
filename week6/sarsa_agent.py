import numpy as np

#-------------TASK 1: SARSA Agent----------------
class SARSA():
    def __init__(self, env, Q, alpha=0.81, gamma=0.96, epsilon=0.9):
        self.env = env
        self.Q = Q
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            #exploration
            return np.random.randint(0, len(self.env.get_posible_actions(state)))
        else:
            #explotation
            return np.argmax(self.Q[state])
        
    def action_function(self, state1, action1, reward, state2, action2):
        #$Q(s,a) \leftarrow (1-\alpha)Q(s,a) + \alpha[R(s) + \gamma Q(s',a')] $
        self.Q[state1][action1] = (1 - self.alpha) * self.Q[state1][action1] + self.alpha * (reward + self.gamma * self.Q[state2][action2])
        
    


# Task 2: Environment cliff-walk 

## Auxiliary function to plot the policy
import numpy as np
import matplotlib.pyplot as plt
def plot_policy(Q, env):
    action_symbols = {
        0: '↑',
        1: '→',
        2: '↓',
        3: '←',
        4: 'E'
    }

    path = get_optimal_path(Q, env)

    fig, ax = plt.subplots()

    for r in range(env.nrows):
        for c in range(env.ncols):

            best_action = np.argmax(Q[r, c])
            if (r, c) == (4, 11):
                best_action = 4  

            symbol = action_symbols[best_action]

            # 🔥 cambiar color si está en el path
            if (r, c) in path:
                color = 'red'
                size = 18
            else:
                color = 'black'
                size = 14

            ax.text(c, env.nrows - r - 1, symbol,
                    ha='center', va='center',
                    fontsize=size, color=color)

    ax.set_xticks(range(env.ncols))
    ax.set_yticks(range(env.nrows))
    ax.grid(True)

    plt.show()


    
def get_optimal_path(Q, env, goal=(4,11), max_steps=100):
    path = [env._find_initial_state()]
    state = env._find_initial_state()

    for _ in range(max_steps):
        r, c = state
        action = np.argmax(Q[r, c])

        if (r, c) == goal:
            break

        # transición manual (ajusta según tu env)
        if action == 0:   # up
            r = max(r-1, 0)
        elif action == 1: # right
            c = min(c+1, env.ncols-1)
        elif action == 2: # down
            r = min(r+1, env.nrows-1)
        elif action == 3: # left
            c = max(c-1, 0)

        state = (r, c)
        path.append(state)

        if state == goal:
            break

    return path


class EnvironmentNuevo:
    def __init__(self, board, exit_state):
        self.board = board
        self.nrows = len(board)
        self.ncols = len(board[0]) 
        self.initial_state = self._find_initial_state()
        self.current_state = self.initial_state
        self.actions = ['up', 'right', 'down', 'left', 'exit']
        # self.P = self._build_transition_matrix()
        self.exit_state = exit_state
        
    def _build_transition_matrix(self):
        nA = len(self.actions)
    
        P = [[[[0 for _ in range(nA)] for _ in range(nA)]
              for _ in range(self.ncols)]
              for _ in range(self.nrows)]
    
        for i in range(self.nrows):
            for j in range(self.ncols):
            
                if self.board[i][j] == '#':
                    continue
                
                # Si es estado terminal
                if self._is_exit(i, j):
                    P[i][j][4][4] = 1.0  # exit → exit
                    continue
                
                for a in range(4):  # solo up,right,down,left
                
                    clockwise = (a + 1) % 4
                    counterclock = (a - 1) % 4
    
                    P[i][j][a][a] = 0.6
                    P[i][j][a][clockwise] = 0.2
                    P[i][j][a][counterclock] = 0.1
                    P[i][j][a][a] += 0.1  # 10% quedarse = misma acción
    
        return P

    def _find_initial_state(self):
        for i in range(self.nrows):
            for j in range(self.ncols):
                if self.board[i][j] == 'S':
                    return (i, j)
        return (1, 1)


    def _is_exit(self, r, c):
        return self.exit_state == (r, c) or self.board[r][c] == '-100'

    def _move(self, r, c, action):
        dr, dc = 0, 0
        if action == 'up':
            dr = -1
        elif action == 'down':
            dr = 1
        elif action == 'left':
            dc = -1
        elif action == 'right':
            dc = 1

        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= self.nrows or nc < 0 or nc >= self.ncols:
            return r, c
        if self.board[nr][nc] == '#':
            return r, c
        return nr, nc
   
    def _get_reward(self, r, c):
        if self._is_exit(r, c):
            return float(self.board[r][c])
        if self.board[r][c] == '#':
            return 0
        if self.board[r][c] == 'S':
            return -1
        return float(self.board[r][c])
    
    def get_current_state(self):
        return self.current_state

    def get_posible_actions(self, state):
        r, c = state
        if self._is_exit(r, c):
            return ['exit']
        if self.board[r][c] == '#':
            return []
        return ['up', 'right', 'down', 'left']
   
    def do_action(self, idx_action):
        r, c = self.current_state
    
        # Si es terminal
        if self._is_exit(r, c):
            if idx_action == 4:
                return float(self.board[r][c]), self.current_state
            return 0, self.current_state
    
        # Elegir acción real 
        real_action = self.actions[idx_action]
    
        nr, nc = self._move(r, c, real_action)
        self.current_state = (nr, nc)
        
    
        reward = self._get_reward(nr, nc)
    
        return reward, self.current_state

    def reset(self):
        self.current_state = self.initial_state

    def is_terminal(self):
        r, c = self.current_state
        return self._is_exit(r, c)
    

if __name__ == "__main__":
    env=EnvironmentNuevo([
    ['-1']*12,
    ['-1']*12,
    ['-1']*12,
    ['-1']*12,
    ['S'] + ['-100']*10 + ['100']
], exit_state=(4, 11))
    
    agent = SARSA(env, Q=np.zeros((env.nrows, env.ncols, len(env.actions))))

    n_episodes = 10000

    for _ in range(n_episodes):
        env.reset()
        state1 = env.get_current_state()
        action1=agent.choose_action(state1)
        # print(f"Episode {_+1}/{n_episodes}")
        while True:
            # if _ == n_episodes - 1:
                # print(f"State: {state1}, Action: {env.actions[action1]}")
                # print("epsilon: ", agent.epsilon)

            reward, state2 = env.do_action(action1)

            action2 = agent.choose_action(state2)
            agent.action_function(state1, action1, reward, state2, action2)

            state1 = env.get_current_state()
            action1 = action2
            if env.is_terminal():
                break
            
        agent.epsilon = agent.epsilon * 0.99  # Decay epsilon for episodes
    plot_policy(agent.Q, env)

 
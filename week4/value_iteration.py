# Enviroment and MDP definition

import random

class EnvironmentNuevo:
    def __init__(self, board):
        self.board = board
        self.nrows = len(board)
        self.ncols = len(board[0]) 
        self.initial_state = self._find_initial_state()
        self.current_state = self.initial_state
        self.actions = ['up', 'right', 'down', 'left', 'exit']
        self.P = self._build_transition_matrix()
        
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
    
        # Elegir acción real según probabilidades
        probs = self.P[r][c][idx_action]
        real_action_idx = random.choices(range(len(self.actions)), weights=probs, k=1)[0]
        real_action = self.actions[real_action_idx]
    
        nr, nc = self._move(r, c, real_action)
        self.current_state = (nr, nc)
    
        if self._is_exit(nr, nc):
            reward = float(self.board[nr][nc])
        else:
            reward = 0
    
        return reward, self.current_state



    def reset(self):
        self.current_state = self.initial_state

    def is_terminal(self):
        r, c = self.current_state
        return self._is_exit(r, c)

    def _is_exit(self, r, c):
        val = self.board[r][c]
        if isinstance(val, str):
            try:
                float(val)
                return True 
            except ValueError:
                return False
        return False

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
   





# ------------- Value Iteration Algorithm for Gridworld -------------
class ValueIteration():
    def __init__(self, env, iterations, gamma=0.9):
        self.mdp = env
        self.discount = gamma
        self.iterations = iterations
        
        # valores como matriz 2D
        self.values = [[0 for _ in range(self.mdp.ncols)]
                       for _ in range(self.mdp.nrows)]

        # usar exactamente las acciones del entorno
        self.actions = self.mdp.actions
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}

    # -------------------------
    # Algoritmo principal
    # -------------------------

    def run_value_iteration(self):
        for _ in range(self.iterations):
            new_values = [row.copy() for row in self.values]

            for r in range(self.mdp.nrows):
                for c in range(self.mdp.ncols):

                    # ignorar paredes
                    if self.mdp.board[r][c] == '#':
                        continue

                    state = (r, c)
                    actions = self.mdp.get_posible_actions(state)

                    # estado terminal
                    if actions == ['exit']:
                        new_values[r][c] = float(self.mdp.board[r][c])
                        continue

                    q_values = []

                    for action in actions:
                        q = self.compute_qvalue_from_values(state, action)
                        q_values.append(q)

                    # print(q_values,actions)
                    if q_values:
                        new_values[r][c] = max(q_values)

            self.values = new_values

    # -------------------------
    # Value function
    # -------------------------

    def get_value(self, state):
        r, c = state
        return self.values[r][c]

    # -------------------------
    # Q-value
    # -------------------------

    def compute_qvalue_from_values(self, state, action):
        r, c = state
        idx_action = self.action_to_idx[action]
    
        q_value = 0
    
        # recorrer acciones reales posibles
        for real_action_idx, prob in enumerate(self.mdp.P[r][c][idx_action]):
        
            if prob == 0:
                continue
            
            real_action = self.actions[real_action_idx]
    
            # transición determinística (NO usar do_action)
            nr, nc = self.mdp._move(r, c, real_action)
    
            reward = 0
            if self.mdp._is_exit(nr, nc):
                reward = float(self.mdp.board[nr][nc])
                q_value += prob * reward
            else:
                next_value = self.values[nr][nc]
                q_value += prob * (reward + self.discount * next_value)

        return q_value


    def get_qvalue(self, state, action):
        return self.compute_qvalue_from_values(state, action)

    # -------------------------
    # Política
    # -------------------------

    def compute_action_from_values(self, state):
        r, c = state

        # si es pared o terminal
        if self.mdp.board[r][c] == '#' or self.mdp._is_exit(r, c):
            return None

        actions = self.mdp.get_posible_actions(state)

        best_action = None
        best_value = float('-inf')

        for action in actions:
            q = self.compute_qvalue_from_values(state, action)
            if q > best_value:
                best_value = q
                best_action = action

        return best_action

    def get_action(self, state):
        return self.compute_action_from_values(state)

    def get_policy(self, state):
        return self.compute_action_from_values(state)

# Graficar la matriz como mapa de calor con valores numéricos

def plot_value_function(values, title="Value Function", save_path=None):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(values)

    for i in range(len(values)):
        for j in range(len(values[0])):
            plt.text(j, i, round(values[i][j], 2),
                     ha='center', va='center')

    plt.title(title)
    plt.xticks([])
    plt.yticks([])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()

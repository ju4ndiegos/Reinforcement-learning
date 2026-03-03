
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
    
                    #MODIFICACION 
                    P[i][j][a][a] = 0.75
                    P[i][j][a][clockwise] = 0.125
                    P[i][j][a][counterclock] = 0.125
    
        return P


    def _find_initial_state(self):
        for i in range(self.nrows):
            for j in range(self.ncols):
                if self.board[i][j] == 'S':
                    return (i, j)
        return (0, 0)

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
   
gridworld = EnvironmentNuevo(
[
['S',' ',' ',' ',' ',' ',' ',' ',' ',' '],
[' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
[' ','#','#','#','#',' ','#','#','#',' '],
[' ',' ',' ',' ','#',' ',' ',' ',' ',' '],
[' ',' ',' ',' ','#','-1',' ',' ',' ',' '],
[' ',' ',' ',' ','#','+1',' ',' ',' ',' '],
[' ',' ',' ',' ','#',' ',' ',' ',' ',' '],
[' ',' ',' ',' ','#','-1','-1',' ',' ',' '],
[' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
[' ',' ',' ',' ',' ',' ',' ',' ',' ',' ']
]
)

import random


class MCM:
    """
    Monte Carlo Control (First-Visit, On-Policy, ε-greedy)
    """

    def __init__(self, env, gamma=0.9, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.env = env
        self.gamma = gamma

        # Parámetros de exploración
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q(s,a)
        self.Q = {}

        # Contador de visitas N(s,a)
        self.N = {}

        # Política greedy
        self.policy = {}

        # Acciones disponibles por estado (precalculadas)
        self.actions_by_state = {}

        self._initialize()

    # ------------------------------------------
    # Inicialización
    # ------------------------------------------
    def _initialize(self):
        for i in range(self.env.nrows):
            for j in range(self.env.ncols):

                if self.env.board[i][j] == '#':
                    continue

                state = (i, j)
                actions = self.env.get_posible_actions(state)

                if not actions:
                    continue

                self.actions_by_state[state] = actions

                for a in actions:
                    self.Q[(state, a)] = 0.0
                    self.N[(state, a)] = 0

                # política inicial aleatoria
                self.policy[state] = random.choice(actions)

    # ------------------------------------------
    # ε-greedy
    # ------------------------------------------
    def _choose_action(self, state):
        actions = self.actions_by_state[state]

        # exploración
        if random.random() < self.epsilon:
            return random.choice(actions)

        # explotación
        q_values = [self.Q[(state, a)] for a in actions]
        max_q = max(q_values)

        best_actions = [a for a in actions if self.Q[(state, a)] == max_q]
        return random.choice(best_actions)

    # ------------------------------------------
    # Generar episodio
    # ------------------------------------------
    def generate_episode(self):
        episode = []

        self.env.reset()
        state = self.env.get_current_state()

        while not self.env.is_terminal():
            action = self._choose_action(state)
            action_idx = self.env.actions.index(action)

            reward, next_state = self.env.do_action(action_idx)

            episode.append((state, action, reward))
            state = next_state

        return episode

    # ------------------------------------------
    # Actualización First-Visit MC
    # ------------------------------------------
    def update(self, episode):

        G = 0
        visited = set()

        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = self.gamma * G + reward

            if (state, action) not in visited:

                visited.add((state, action))

                # Incrementar contador
                self.N[(state, action)] += 1

                # Promedio incremental
                alpha = 1 / self.N[(state, action)]
                self.Q[(state, action)] += alpha * (G - self.Q[(state, action)])

                # Mejora de política (greedy)
                actions = self.actions_by_state[state]
                q_values = [self.Q[(state, a)] for a in actions]
                max_q = max(q_values)
                best_actions = [a for a in actions if self.Q[(state, a)] == max_q]
                self.policy[state] = random.choice(best_actions)

    # ------------------------------------------
    # Entrenamiento
    # ------------------------------------------
    def train(self, max_episodes=10000, stability_window=200):
        stable_count = 0
        prev_policy = None
    
        for episode in range(1, max_episodes + 1):
        
            # GLIE decay
            self.epsilon = 1 / (episode ** 0.5)
    
            episode_data = self.generate_episode()
            self.update(episode_data)
    
            # Chequeo de estabilidad de política
            current_policy = self.policy.copy()
    
            if current_policy == prev_policy:
                stable_count += 1
            else:
                stable_count = 0
    
            if stable_count >= stability_window:
                print(f"Convergió en episodio {episode}")
                return
    
            prev_policy = current_policy
    
        print("Entrenamiento terminó sin convergencia detectada")


    # ------------------------------------------
    # Obtener política final
    # ------------------------------------------
    def get_policy(self):
        return self.policy

    # ------------------------------------------
    # Obtener valores V(s)
    # ------------------------------------------
    def get_state_values(self):
        V = {}

        for state, actions in self.actions_by_state.items():
            V[state] = max(self.Q[(state, a)] for a in actions)

        return V

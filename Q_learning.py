# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import random

# class MovingObject:
#     def __init__(self, size=5):
#         self.size = size
#         self.object_pos = [0, np.random.randint(0, size-1)]  # Sempre in riga 0

#     def step(self):
#         """Muove l'oggetto verso il basso"""
#         if self.object_pos[0] < self.size - 1:
#             self.object_pos[0] += 1  # Si muove verso il basso
#         else:
#             self.reset_position()  # Se arriva in fondo, riparte dalla riga 0

#     def reset_position(self):
#         """Resetta il puck esattamente in cima"""
#         self.object_pos = [0, np.random.randint(0, self.size)]  # Sempre in riga 0

# class Agent:
#     def __init__(self, size=5):
#         """Definisce posizioni valide dell'agente e sceglie una posizione iniziale"""
#         self.valid_positions = [(size - 1, 2), (size - 2, 1), (size - 2, 3), (size-3, 0), (size - 3, 4)]  # Posizioni in basso
#         self.position = self.valid_positions[0]  # Posizione iniziale casuale


#     def move(self):
#         """Muove l'agente in una nuova posizione valida"""
#         self.position = random.choice(self.valid_positions)

# # Parametri
# size = 5  # Dimensione della griglia
# num_simulations = 3  # Numero totale di simulazioni
# Tmax = 5  # Durata totale in secondi
# interval = (Tmax / size) * 1000  # Durata di ogni frame in ms

# # Creazione della figura
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
# ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
# ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
# ax.set_facecolor("white")
# ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

# ax.invert_yaxis()  # Inverti l'asse Y per correggere l'orientamento

# # Inizializza gli elementi dell'animazione
# puck, = ax.plot([], [], 'bo', markersize=15)  # Puck (blu)
# mallet, = ax.plot([], [], 'ro', markersize=25)  # Agente (rosso)
# hit_text = ax.text(2, 2, "", fontsize=20, color="black", ha='center', va='center')

# def init():
#     """Mostra lo stato iniziale del puck e mallet"""
#     puck.set_data([], [])  # Inizializza il puck
#     mallet.set_data([], [])  # Inizializza il mallet
#     hit_text.set_text("")
#     return puck, mallet, hit_text

# def update(frame, env, agent):
#     """Aggiorna la posizione dell'oggetto e controlla la collisione"""
#     if frame > 0:  # Evita che il primo frame sposti subito il puck
#         env.step()
#         agent.move()  # Muove l'agente in una nuova posizione ad ogni timestep

#     puck.set_data([env.object_pos[1]], [env.object_pos[0]])
#     mallet.set_data([agent.position[1]], [agent.position[0]])

#     # Controlla collisione
#     if env.object_pos == list(agent.position):
#         hit_text.set_text("HIT")
#         print("HIT! Simulation stopped.")
#         return puck, mallet, hit_text

#     return puck, mallet, hit_text

# env = MovingObject(size)
# agent = Agent(size)
# action_space_size = len(agent.valid_positions)
# state_space_size = size*size
# q_table = np.zeros((state_space_size, action_space_size))

# for simulation_num in range(num_simulations):
#     # Creazione dell'ambiente e dell'agente per ogni simulazione
#     env = MovingObject(size)
#     agent = Agent(size)
    
#     # Inizializza la figura per la nuova simulazione
#     init()

#     # Esegui l'animazione
#     ani = animation.FuncAnimation(fig, update, frames=size, fargs=(env, agent), interval=interval, init_func=init, repeat=False)

#     # Stampa le posizioni iniziali
#     # print(f"Simulation {simulation_num + 1} starting:")
#     # print(f"Puck position: {env.object_pos}")
#     # print(f"Mallet position: {agent.position}")

#     # Esegui la simulazione per il tempo massimo o fino a un colpo
#     for _ in range(size):
#         plt.pause(interval / 1000)  # Pausa per il tempo di ogni frame
#         if hit_text.get_text() == "HIT":
#             break  # Esci dal ciclo se c'è stato un colpo

























# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import random
# from tqdm import tqdm
# import matplotlib.patches as patches

# # Funzione di utilità per convertire la posizione [riga, colonna] in un indice intero
# def state_to_index(state, size):
#     return state[0] * size + state[1]

# class MovingObject:
#     def __init__(self, size=5):
#         self.size = size
#         # Il puck parte sempre dalla riga 0 e da una colonna casuale
#         self.object_pos = [0, np.random.randint(0, size)]
    
#     def step(self):
#         """Muove l'oggetto verso il basso"""
#         if self.object_pos[0] < self.size - 1:
#             self.object_pos[0] += 1  # Si muove verso il basso
#         else:
#             self.reset_position()  # Se arriva in fondo, riparte dalla riga 0
    
#     def reset_position(self):
#         """Resetta il puck esattamente in cima"""
#         self.object_pos = [0, np.random.randint(0, self.size)]

# class Agent:
#     def __init__(self, size=5):
#         """
#         Definisce le posizioni valide dell'agente e sceglie una posizione iniziale.
#         Le posizioni sono scelte in base a una logica di posizionamento in basso.
#         """
#         self.valid_positions = [
#             (size - 1, 2),
#             (size - 2, 1),
#             (size - 2, 3),
#             (size - 3, 0),
#             (size - 3, 4)
#         ]
#         self.position = self.valid_positions[0]  # Posizione iniziale

#     def move(self, action):
#         """L'agente si muove in una delle posizioni valide, in base all'azione scelta."""
#         self.position = self.valid_positions[action]

# # Parametri della simulazione e dell'algoritmo Q-learning
# size = 5                   # Dimensione della griglia
# num_episodes = 10000    # Numero totale di episodi
# max_steps_per_episode = 5  # Numero massimo di step per episodio

# learning_rate = 0.1
# discount_rate = 0.995

# exploration_rate = 1
# max_exploration_rate = 1
# min_exploration_rate = 0.001
# exploration_decay_rate = 0.001

# # Creazione della figura per la visualizzazione (opzionale)
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
# ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
# ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
# ax.set_facecolor("white")
# ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
# ax.invert_yaxis()  # Inverti l'asse Y per correggere l'orientamento

# # Inizializza gli elementi grafici
# puck, = ax.plot([], [], 'bo', markersize=15)    # Puck (blu)
# mallet, = ax.plot([], [], 'ro', markersize=25)    # Agente (rosso)
# hit_text = ax.text(2, 2, "", fontsize=20, color="black", ha='center', va='center')

# def init():
#     """Mostra lo stato iniziale del puck e dell'agente"""
#     puck.set_data([], [])
#     mallet.set_data([], [])
#     hit_text.set_text("")
#     return puck, mallet, hit_text

# def compute_reward(env, agent, action):
#     # Estrai le coordinate del puck e dell'agente
#     puck_row, puck_col = env.object_pos
#     agent_row, agent_col = agent.position
    
#     # Calcola la distanza verticale e orizzontale
#     vertical_distance = agent_row - puck_row
#     horizontal_distance = abs(agent_col - puck_col)
    
#     # Caso di HIT: l'agente intercetta il puck
#     if env.object_pos == list(agent.position):
#         return 100  # Premio elevato per il colpo riuscito
    
#     # Caso 1: Il puck è ancora lontano (più di 2 righe sopra l'agente)
#     if vertical_distance >= 2:
#         # Se l'agente è già allineato orizzontalmente, il reward per stare fermo può essere un po' più alto
#         if action == 0:
#             if horizontal_distance == 0:
#                 return 50  # Ottimo posizionamento, aspetta!
#             else:
#                 return -70  # Aspetta, ma non è perfettamente allineato
#         else:
#             # Penalizza fortemente il movimento prematuro
#             return -50
        
#     if vertical_distance < 2: 
#         if action == 0:
#             return 60
#         else:
#             return -60


    




# def update(frame, env, agent, action, done):
#     """
#     Aggiorna la posizione dell'oggetto e controlla la collisione.
#     Restituisce: (stato, reward, done)
#     """
#     if frame > 0:
#         env.step()
#         agent.move(action)
    
#     # Aggiorna la posizione grafica
#     puck.set_data([env.object_pos[1]], [env.object_pos[0]])
#     mallet.set_data([agent.position[1]], [agent.position[0]])
    
#     # Controlla collisione: se il puck e l'agente coincidono -> HIT
#     reward = compute_reward(env, agent, action)
#     if env.object_pos == list(agent.position):
#         hit_text.set_text("HIT")
#         done = True

#     state = env.object_pos
#     return state, reward, done

# # Inizializza ambiente, agente, Q-table, ecc.
# env = MovingObject(size)
# agent = Agent(size)
# action_space_size = len(agent.valid_positions)
# state_space_size = size * size
# q_table = np.zeros((state_space_size, action_space_size))
# rewards_all_episodes = []

# saved_trajectories = []  # Ogni elemento sarà un dizionario: {"episode": ..., "puck": [...], "agent": [...]}
# successes = []  # per ogni episodio, 1 se HIT, 0 altrimenti
# success_rates = []  # percentuale di successo ogni 1000 episodi

# # Q-learning algorithm
# for episode in tqdm(range(num_episodes), desc="Training episodes"):
#     # Ogni episodio inizia con un nuovo ambiente
#     env = MovingObject(size)
#     init()
#     state = env.object_pos
#     state_index = state_to_index(state, size)
#     done = False
#     rewards_current_episode = 0

#     # Inizializza le liste per registrare la traiettoria dell'episodio corrente
#     puck_trajectory = []
#     agent_trajectory = []
#     episode_success = 0

#     for step in range(max_steps_per_episode):
#         # Trade-off tra esplorazione e sfruttamento
#         exploration_rate_threshold = random.uniform(0, 1)
#         if exploration_rate_threshold > exploration_rate:
#             action = np.argmax(q_table[state_index, :])
#         else:
#             action = random.randint(0, action_space_size - 1)
        
#         new_state, reward, done = update(step, env, agent, action, done)
#         new_state_index = state_to_index(new_state, size)
#         # Aggiornamento della Q-table
#         q_table[state_index, action] = q_table[state_index, action] * (1 - learning_rate) + \
#             learning_rate * (reward + discount_rate * np.max(q_table[new_state_index, :]))
        
#         state_index = new_state_index
#         rewards_current_episode += reward

#         # Registra le posizioni correnti (usa .copy() per evitare riferimenti)
#         puck_trajectory.append(env.object_pos.copy())
#         agent_trajectory.append(agent.position)
        
#         if done:
#             episode_success = 1  # HIT avvenuto
#             break



#     successes.append(episode_success)

#     # Aggiornamento (decadimento) dell'exploration rate
#     exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

#     rewards_all_episodes.append(rewards_current_episode)

#     if (episode + 1) % 1000 == 0:
#         recent_success_rate = sum(successes[-1000:]) / 1000 * 100  # in percentuale
#         success_rates.append(recent_success_rate)
    
#     # Se l'episodio è maggiore di 90000 e si è verificato un HIT, salva la traiettoria
#     if episode > (num_episodes*(0.9)) and done:
#         print(f"Episode {episode}: Reward = {rewards_current_episode}")
#         print("Puck position:", env.object_pos)
#         print("Agent position:", agent.position)
#         saved_trajectories.append({
#             "episode": episode,
#             "puck": puck_trajectory.copy(),
#             "agent": agent_trajectory.copy()
#         })

# # Se abbiamo salvato più di 10 traiettorie, prendi le ultime 10
# if len(saved_trajectories) > 10:
#     saved_trajectories = saved_trajectories[-10:]

# # Genera e salva i plot delle traiettorie
# for traj in saved_trajectories:
#     plt.figure(figsize=(5, 5))
#     puck_traj = np.array(traj["puck"])
#     agent_traj = np.array(traj["agent"])
#     # Plot della traiettoria del puck (linea blu con marker)
#     plt.plot(puck_traj[:, 1], puck_traj[:, 0], 'bo-', label="Puck trajectory")
#     # Plot della traiettoria dell'agente (linea rossa con marker)
#     plt.plot(agent_traj[:, 1], agent_traj[:, 0], 'ro-', label="Agent trajectory")
#     plt.title(f"Trajectory - Episode {traj['episode']}")
#     plt.xlabel("Colonna")
#     plt.ylabel("Riga")
#     plt.legend()
#     plt.gca().invert_yaxis()  # Per avere la riga 0 in alto
#     plt.savefig(f"trajectory_episode_{traj['episode']}.png")
#     plt.close()

# # Calcola e stampa il reward medio ogni mille episodi
# rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes/1000)
# count = 1000
# print("******** Average reward per thousand episodes ********\n")
# for r in rewards_per_thousand_episodes:
#     print(f"{count}: {sum(r)/1000}")
#     count += 1000

# # Stampa la Q-table finale
# print("\n\n******** Q-table ********\n")
# print(q_table)

# def plot_colored_trajectory(traj, grid_size=5):
#     """
#     Visualizza la traiettoria del puck e dell'agente colorando l'intera cella corrispondente a ciascun step.
#     Se le posizioni coincidono (HIT), la cella viene colorata in verde.
#     Viene inoltre visualizzato il numero totale di azioni compiute dall'agente.
#     """
#     fig, ax = plt.subplots(figsize=(5, 5))
    
#     # Imposta la griglia
#     ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
#     ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
#     ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
#     ax.set_xlim(-0.5, grid_size - 0.5)
#     ax.set_ylim(grid_size - 0.5, -0.5)
#     ax.set_xticks(np.arange(grid_size))
#     ax.set_yticks(np.arange(grid_size))
#     ax.set_xticklabels(np.arange(grid_size))
#     ax.set_yticklabels(np.arange(grid_size))
    
#     # Numero totale di azioni (step) fatti dall'agente
#     total_actions = len(traj["agent"])
#     ax.set_title(f"Episode {traj['episode']} - Totale azioni agente: {total_actions}")
    
#     # Per ogni step della traiettoria, colora la cella e inserisci una label
#     for i, (puck_pos, agent_pos) in enumerate(zip(traj["puck"], traj["agent"])):
#         # Estrai le coordinate (riga, colonna)
#         puck_r, puck_c = puck_pos
#         agent_r, agent_c = agent_pos
        
#         # Se le posizioni coincidono, segnala HIT
#         if puck_pos == list(agent_pos):
#             # Cella verde per il colpo (HIT)
#             rect = patches.Rectangle((agent_c - 0.5, agent_r - 0.5), 1, 1,
#                                      facecolor="green", alpha=0.7, edgecolor="black")
#             ax.add_patch(rect)
#             ax.text(agent_c, agent_r, f"HIT\n({i})", color="white",
#                     ha="center", va="center", fontsize=10, weight="bold")
#         else:
#             # Cella blu per il puck
#             rect_puck = patches.Rectangle((puck_c - 0.5, puck_r - 0.5), 1, 1,
#                                           facecolor="blue", alpha=0.5, edgecolor="black")
#             ax.add_patch(rect_puck)
#             ax.text(puck_c, puck_r, f"P{i}", color="white",
#                     ha="center", va="center", fontsize=10, weight="bold")
            
#             # Cella rossa per l'agente
#             rect_agent = patches.Rectangle((agent_c - 0.5, agent_r - 0.5), 1, 1,
#                                            facecolor="red", alpha=0.5, edgecolor="black")
#             ax.add_patch(rect_agent)
#             ax.text(agent_c, agent_r, f"A{i}", color="white",
#                     ha="center", va="center", fontsize=10, weight="bold")
    
#     plt.savefig(f"trajectory_episode_{traj['episode']}.png")
#     plt.close()

# for traj in saved_trajectories:
#     plot_colored_trajectory(traj, grid_size=size)


# plt.figure(figsize=(10, 5))
# plt.plot(np.arange(1000, num_episodes + 1, 1000), success_rates, color='green', linewidth=2)
# plt.xlabel('Episodi')
# plt.ylabel('Percentuale di Successo (%)')
# plt.title('Andamento della Percentuale di Successo durante l\'addestramento')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()




#VERTICAL
# import numpy as np
# import matplotlib.pyplot as plt
# import random
# from tqdm import tqdm
# import matplotlib.patches as patches

# # Funzione per convertire lo stato composto (puck e agente) in un indice univoco.
# def state_to_index(puck_state, agent_state, grid_size, num_agent_positions):
#     # puck_state: [riga, colonna]
#     # agent_state: indice (0,...,num_agent_positions-1)
#     puck_index = puck_state[0] * grid_size + puck_state[1]
#     return puck_index * num_agent_positions + agent_state

# # Ambiente: il puck si muove verso il basso.
# class MovingObject:
#     def __init__(self, size=7):
#         self.size = size
#         # Il puck parte dalla riga 0 e da una colonna casuale
#         self.object_pos = [0, np.random.randint(0, size)]
    
#     def step(self):
#         if self.object_pos[0] < self.size - 1:
#             self.object_pos[0] += 1  # si muove verso il basso
#         else:
#             self.reset_position()   # se arriva in fondo, riparte dalla cima
    
#     def reset_position(self):
#         self.object_pos = [0, np.random.randint(0, self.size)]

# # Agente: può scegliere tra 7 posizioni valide.
# # Le posizioni sono state definite per la griglia 7x7:
# # - La posizione di "attesa" (waiting) è in fondo, scelta come centro: (6,3)
# # - Le posizioni intermedie sono nelle righe superiori, distribuite simmetricamente.
# class Agent:
#     def __init__(self, size=7):
#         self.valid_positions = [
#             (size - 1, 3),  # Azione 0: waiting – posizione ideale se il puck arriva in riga 6
#             (size - 2, 2),  # Azione 1: riga 5, a sinistra del centro
#             (size - 2, 4),  # Azione 2: riga 5, a destra del centro
#             (size - 3, 1),  # Azione 3: riga 4, diagonale sinistra
#             (size - 3, 5),  # Azione 4: riga 4, diagonale destra
#             (size - 4, 0),  # Azione 5: riga 3, ulteriore diagonale a sinistra
#             (size - 4, 6)   # Azione 6: riga 3, ulteriore diagonale a destra
#         ]
#         self.position = self.valid_positions[0]  # posizione iniziale
    
#     def move(self, action):
#         # L'agente si sposta nella posizione valida corrispondente all'azione scelta.
#         self.position = self.valid_positions[action]

# # Funzione che restituisce lo stato corrente come indice intero.
# def get_state(env, agent, grid_size):
#     puck_state = env.object_pos
#     agent_state = agent.valid_positions.index(agent.position)
#     num_agent_positions = len(agent.valid_positions)
#     return state_to_index(puck_state, agent_state, grid_size, num_agent_positions)

# # Per ogni posizione valida dell'agente, definiamo la "target row"
# # in cui il puck dovrà trovarsi per un colpo ottimale.
# def get_target_row(agent_position):
#     mapping = {
#         (6, 3): 6,  # waiting: colpire quando il puck è in riga 6
#         (5, 2): 5,
#         (5, 4): 5,
#         (4, 1): 4,
#         (4, 5): 4,
#         (3, 0): 3,
#         (3, 6): 3
#     }
#     return mapping[agent_position]

# # Funzione di reward rivista.
# def compute_reward(env, agent):
#     puck_r, puck_c = env.object_pos
#     agent_r, agent_c = agent.position
#     # Se c'è HIT:
#     if [puck_r, puck_c] == list(agent.position):
#         return 100
    
#     # Determina la riga target per il colpo in base alla posizione scelta dall'agente
#     target_row = get_target_row(agent.position)
#     error = target_row - puck_r  # quanti step mancano al raggiungimento della target row
#     horizontal_distance = abs(agent_c - puck_c)
    
#     # Se l'agente è allineato orizzontalmente, il reward aumenta man mano che il puck si avvicina alla target row.
#     if horizontal_distance == 0:
#         if error > 0:  # il puck non è ancora arrivato alla target row
#             return 50 - 10 * error  # error minore → reward maggiore
#         elif error == 0:
#             # Se il puck è esattamente nella target row (ma non ancora in collisione)
#             return 80
#         else:  # il puck ha superato la target row
#             return -100 - 10 * abs(error)
#     else:
#         # Se l'agente non è allineato, penalizziamo in base alla distanza orizzontale.
#         return -50 - 10 * horizontal_distance

# # Parametri di simulazione e Q-learning
# size = 7                   # griglia 7x7
# num_episodes = 10000       # numero totale di episodi
# max_steps_per_episode = size  # numero massimo di step per episodio

# learning_rate = 0.1
# discount_rate = 0.995

# exploration_rate = 1
# max_exploration_rate = 1
# min_exploration_rate = 0.001
# exploration_decay_rate = 0.001

# # Impostazione della figura per la visualizzazione
# fig, ax = plt.subplots(figsize=(7, 7))
# ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
# ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
# ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
# ax.set_facecolor("white")
# ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
# ax.invert_yaxis()

# puck_plot, = ax.plot([], [], 'bo', markersize=15)
# mallet_plot, = ax.plot([], [], 'ro', markersize=25)
# hit_text = ax.text(size/2, size/2, "", fontsize=20, color="black", ha='center', va='center')

# def init():
#     puck_plot.set_data([], [])
#     mallet_plot.set_data([], [])
#     hit_text.set_text("")
#     return puck_plot, mallet_plot, hit_text

# def update(frame, env, agent, action, done):
#     if frame > 0:
#         env.step()
#         agent.move(action)
#     puck_plot.set_data([env.object_pos[1]], [env.object_pos[0]])
#     mallet_plot.set_data([agent.position[1]], [agent.position[0]])
#     reward = compute_reward(env, agent)
#     if [env.object_pos[0], env.object_pos[1]] == list(agent.position):
#         hit_text.set_text("HIT")
#         done = True
#     state = (env.object_pos.copy(), agent.position)
#     return state, reward, done

# # Inizializza ambiente, agente e Q-table.
# env = MovingObject(size)
# agent = Agent(size)
# action_space_size = len(agent.valid_positions)   # 7 azioni
# num_agent_positions = len(agent.valid_positions)
# state_space_size = size * size * num_agent_positions  # 7*7*7 = 343
# q_table = np.zeros((state_space_size, action_space_size))
# rewards_all_episodes = []

# saved_trajectories = []  # per visualizzare alcune traiettorie
# successes = []          # 1 se HIT, 0 altrimenti
# success_rates = []      # percentuale di successo ogni 1000 episodi

# # Ciclo di Q-learning
# for episode in tqdm(range(num_episodes), desc="Training episodes"):
#     env = MovingObject(size)
#     agent = Agent(size)  # reset dell'agente alla posizione iniziale (waiting)
#     init()
#     state_index = get_state(env, agent, size)
#     done = False
#     rewards_current_episode = 0
#     puck_trajectory = []
#     agent_trajectory = []
#     episode_success = 0
    
#     for step in range(max_steps_per_episode):
#         exploration_rate_threshold = random.uniform(0, 1)
#         if exploration_rate_threshold > exploration_rate:
#             action = np.argmax(q_table[state_index, :])
#         else:
#             action = random.randint(0, action_space_size - 1)
        
#         new_state, reward, done = update(step, env, agent, action, done)
#         new_state_index = get_state(env, agent, size)
        
#         q_table[state_index, action] = q_table[state_index, action] * (1 - learning_rate) + \
#             learning_rate * (reward + discount_rate * np.max(q_table[new_state_index, :]))
        
#         state_index = new_state_index
#         rewards_current_episode += reward
        
#         puck_trajectory.append(env.object_pos.copy())
#         agent_trajectory.append(agent.position)
        
#         if done:
#             episode_success = 1
#             break
    
#     successes.append(episode_success)
#     # Aggiorna l'exploration rate
#     exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
#     rewards_all_episodes.append(rewards_current_episode)
    
#     if (episode + 1) % 1000 == 0:
#         recent_success_rate = sum(successes[-1000:]) / 1000 * 100
#         success_rates.append(recent_success_rate)
    
#     # Salva alcune traiettorie negli ultimi episodi (ultimo 10%) se c'è HIT
#     if episode > (num_episodes * 0.9) and done:
#         print(f"Episode {episode}: Reward = {rewards_current_episode}")
#         print("Puck position:", env.object_pos)
#         print("Agent position:", agent.position)
#         saved_trajectories.append({
#             "episode": episode,
#             "puck": puck_trajectory.copy(),
#             "agent": agent_trajectory.copy()
#         })

# if len(saved_trajectories) > 10:
#     saved_trajectories = saved_trajectories[-10:]

# # Genera e salva i plot delle traiettorie salvate.
# for traj in saved_trajectories:
#     plt.figure(figsize=(7, 7))
#     puck_traj = np.array(traj["puck"])
#     agent_traj = np.array(traj["agent"])
#     plt.plot(puck_traj[:, 1], puck_traj[:, 0], 'bo-', label="Puck trajectory")
#     plt.plot(agent_traj[:, 1], agent_traj[:, 0], 'ro-', label="Agent trajectory")
#     plt.title(f"Trajectory - Episode {traj['episode']}")
#     plt.xlabel("Colonna")
#     plt.ylabel("Riga")
#     plt.legend()
#     plt.gca().invert_yaxis()
#     plt.savefig(f"trajectory_episode_{traj['episode']}.png")
#     plt.close()

# # Stampa il reward medio ogni mille episodi
# rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
# count = 1000
# print("******** Average reward per thousand episodes ********\n")
# for r in rewards_per_thousand_episodes:
#     print(f"{count}: {sum(r) / 1000}")
#     count += 1000

# print("\n\n******** Q-table ********\n")
# print(q_table)

# # Funzione di visualizzazione colorata delle traiettorie
# def plot_colored_trajectory(traj, grid_size=7):
#     fig, ax = plt.subplots(figsize=(7, 7))
    
#     ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
#     ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
#     ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
#     ax.set_xlim(-0.5, grid_size - 0.5)
#     ax.set_ylim(grid_size - 0.5, -0.5)
#     ax.set_xticks(np.arange(grid_size))
#     ax.set_yticks(np.arange(grid_size))
#     ax.set_xticklabels(np.arange(grid_size))
#     ax.set_yticklabels(np.arange(grid_size))
    
#     total_actions = len(traj["agent"])
#     ax.set_title(f"Episode {traj['episode']} - Totale azioni agente: {total_actions}")
    
#     for i, (puck_pos, agent_pos) in enumerate(zip(traj["puck"], traj["agent"])):
#         puck_r, puck_c = puck_pos
#         agent_r, agent_c = agent_pos
        
#         if puck_pos == list(agent_pos):
#             rect = patches.Rectangle((agent_c - 0.5, agent_r - 0.5), 1, 1,
#                                      facecolor="green", alpha=0.7, edgecolor="black")
#             ax.add_patch(rect)
#             ax.text(agent_c, agent_r, f"HIT\n({i})", color="white",
#                     ha="center", va="center", fontsize=10, weight="bold")
#         else:
#             rect_puck = patches.Rectangle((puck_c - 0.5, puck_r - 0.5), 1, 1,
#                                           facecolor="blue", alpha=0.5, edgecolor="black")
#             ax.add_patch(rect_puck)
#             ax.text(puck_c, puck_r, f"P{i}", color="white",
#                     ha="center", va="center", fontsize=10, weight="bold")
            
#             rect_agent = patches.Rectangle((agent_c - 0.5, agent_r - 0.5), 1, 1,
#                                            facecolor="red", alpha=0.5, edgecolor="black")
#             ax.add_patch(rect_agent)
#             ax.text(agent_c, agent_r, f"A{i}", color="white",
#                     ha="center", va="center", fontsize=10, weight="bold")
    
#     plt.savefig(f"trajectory_episode_{traj['episode']}.png")
#     plt.close()

# for traj in saved_trajectories:
#     plot_colored_trajectory(traj, grid_size=size)

# plt.figure(figsize=(10, 5))
# plt.plot(np.arange(1000, num_episodes + 1, 1000), success_rates, color='green', linewidth=2)
# plt.xlabel('Episodi')
# plt.ylabel('Percentuale di Successo (%)')
# plt.title("Andamento della Percentuale di Successo durante l'addestramento")
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import matplotlib.patches as patches

# Funzione per convertire lo stato composto (puck e agente) in un indice univoco.
def state_to_index(puck_state, agent_state, grid_size, num_agent_positions):
    # puck_state: [riga, colonna]
    # agent_state: indice (0,...,num_agent_positions-1)
    puck_index = puck_state[0] * grid_size + puck_state[1]
    return puck_index * num_agent_positions + agent_state

# Ambiente: il puck si muove verso il basso.
class MovingObject:
    def __init__(self, size=7):
        self.size = size
        self.object_pos = [0, np.random.randint(0, size)]
    
    def step(self, episode, num_episodes):
        # Check if we are past 70% of training
        if episode > 0.7 * num_episodes:
            # Move diagonally: down-left (-1) or down-right (+1)
            diagonal_shift = random.choice([-1, 1])
            new_col = self.object_pos[1] + diagonal_shift
            # Keep within grid boundaries
            new_col = max(0, min(self.size - 1, new_col))
            self.object_pos[1] = new_col  # Apply horizontal movement
        
        # Move down (this happens always)
        if self.object_pos[0] < self.size - 1:
            self.object_pos[0] += 1
        else:
            self.reset_position()
    
    def reset_position(self):
        self.object_pos = [0, np.random.randint(0, self.size)]

# Agente: può scegliere tra 7 posizioni valide.
# Le posizioni sono state definite per la griglia 7x7:
# - La posizione di "attesa" (waiting) è in fondo, scelta come centro: (6,3)
# - Le posizioni intermedie sono nelle righe superiori, distribuite simmetricamente.
class Agent:
    def __init__(self, size=7):
        self.valid_positions = [
            (size - 1, 3),  # Azione 0: waiting – posizione ideale se il puck arriva in riga 6
            (size - 2, 2),  # Azione 1: riga 5, a sinistra del centro
            (size - 2, 4),  # Azione 2: riga 5, a destra del centro
            (size - 3, 1),  # Azione 3: riga 4, diagonale sinistra
            (size - 3, 5),  # Azione 4: riga 4, diagonale destra
            (size - 4, 0),  # Azione 5: riga 3, ulteriore diagonale a sinistra
            (size - 4, 6)   # Azione 6: riga 3, ulteriore diagonale a destra
        ]
        self.position = self.valid_positions[0]  # posizione iniziale
    
    def move(self, action):
        # L'agente si sposta nella posizione valida corrispondente all'azione scelta.
        self.position = self.valid_positions[action]

# Funzione che restituisce lo stato corrente come indice intero.
def get_state(env, agent, grid_size):
    puck_state = env.object_pos
    agent_state = agent.valid_positions.index(agent.position)
    num_agent_positions = len(agent.valid_positions)
    return state_to_index(puck_state, agent_state, grid_size, num_agent_positions)

# Per ogni posizione valida dell'agente, definiamo la "target row"
# in cui il puck dovrà trovarsi per un colpo ottimale.
def get_target_row(agent_position):
    mapping = {
        (6, 3): 6,  # waiting: colpire quando il puck è in riga 6
        (5, 2): 5,
        (5, 4): 5,
        (4, 1): 4,
        (4, 5): 4,
        (3, 0): 3,
        (3, 6): 3
    }
    return mapping[agent_position]

# Funzione di reward rivista.
def compute_reward(env, agent):
    puck_r, puck_c = env.object_pos
    agent_r, agent_c = agent.position

    # Se c'è HIT
    if (puck_r, puck_c) == (agent_r, agent_c):
        return 100  # Massima ricompensa

    target_row = get_target_row(agent.position)
    row_error = target_row - puck_r  # Distanza dalla target row
    col_error = abs(agent_c - puck_c)  # Distanza orizzontale

    # Penalità per mancato allineamento orizzontale
    horizontal_penalty = -10 * col_error 

    # Premiamo se l'agente è già allineato orizzontalmente e il puck non ha ancora superato la target row
    if col_error == 0:
        if row_error > 0:
            return 50 - 10 * row_error  # Più vicino → reward maggiore
        elif row_error == 0:
            return 80  # Premiamo se è in target row ma non ha ancora colpito
        else:
            return -100 - 10 * abs(row_error)  # Penalizziamo se ha superato

    # Se il puck si muove diagonalmente, diamo un leggero premio se l'agente prevede la mossa
    if env.object_pos[0] > 0 and abs(env.object_pos[1] - puck_c) == 1:
        prediction_bonus = 20
    else:
        prediction_bonus = 0

    return horizontal_penalty + prediction_bonus


# Parametri di simulazione e Q-learning
size = 7                   # griglia 7x7
num_episodes = 500000      # numero totale di episodi
max_steps_per_episode = size  # numero massimo di step per episodio

learning_rate = 0.1
discount_rate = 0.995

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.001
exploration_decay_rate = 0.0005 

# Impostazione della figura per la visualizzazione
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
ax.set_facecolor("white")
ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
ax.invert_yaxis()

puck_plot, = ax.plot([], [], 'bo', markersize=15)
mallet_plot, = ax.plot([], [], 'ro', markersize=25)
hit_text = ax.text(size/2, size/2, "", fontsize=20, color="black", ha='center', va='center')

def init():
    puck_plot.set_data([], [])
    mallet_plot.set_data([], [])
    hit_text.set_text("")
    return puck_plot, mallet_plot, hit_text

def update(frame, env, agent, action, done, episode, num_episodes):
    if frame > 0:
        env.step(episode, num_episodes)  # Pass episode number
        agent.move(action)
    puck_plot.set_data([env.object_pos[1]], [env.object_pos[0]])
    mallet_plot.set_data([agent.position[1]], [agent.position[0]])
    reward = compute_reward(env, agent)
    if [env.object_pos[0], env.object_pos[1]] == list(agent.position):
        hit_text.set_text("HIT")
        done = True
    state = (env.object_pos.copy(), agent.position)
    return state, reward, done

# Inizializza ambiente, agente e Q-table.
env = MovingObject(size)
agent = Agent(size)
action_space_size = len(agent.valid_positions)   # 7 azioni
num_agent_positions = len(agent.valid_positions)
state_space_size = size * size * num_agent_positions  # 7*7*7 = 343
q_table = np.random.uniform(low=-10, high=10, size=(state_space_size, action_space_size))
rewards_all_episodes = []

saved_trajectories = []  # per visualizzare alcune traiettorie
successes = []          # 1 se HIT, 0 altrimenti
success_rates = []      # percentuale di successo ogni 1000 episodi

# Ciclo di Q-learning
for episode in tqdm(range(num_episodes), desc="Training episodes"):
    env = MovingObject(size)
    agent = Agent(size)  # reset dell'agente alla posizione iniziale (waiting)
    init()
    state_index = get_state(env, agent, size)
    done = False
    rewards_current_episode = 0
    puck_trajectory = []
    agent_trajectory = []
    episode_success = 0
    learning_rate = max(0.1 * (0.99 ** episode), 0.01)
    
    for step in range(max_steps_per_episode):
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state_index, :])
        else:
            action = random.randint(0, action_space_size - 1)
        
        new_state, reward, done = update(step, env, agent, action, done, episode, num_episodes)

        new_state_index = get_state(env, agent, size)
        
        q_table[state_index, action] = q_table[state_index, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state_index, :]))
        
        state_index = new_state_index
        rewards_current_episode += reward
        
        puck_trajectory.append(env.object_pos.copy())
        agent_trajectory.append(agent.position)
        
        if done:
            episode_success = 1
            break
    
    successes.append(episode_success)
    # Aggiorna l'exploration rate
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
    rewards_all_episodes.append(rewards_current_episode)
    
    if (episode + 1) % 1000 == 0:
        recent_success_rate = sum(successes[-1000:]) / 1000 * 100
        success_rates.append(recent_success_rate)
    
    # Salva alcune traiettorie negli ultimi episodi (ultimo 10%) se c'è HIT
    if episode > (num_episodes * 0.9) and done:
        print(f"Episode {episode}: Reward = {rewards_current_episode}")
        print("Puck position:", env.object_pos)
        print("Agent position:", agent.position)
        saved_trajectories.append({
            "episode": episode,
            "puck": puck_trajectory.copy(),
            "agent": agent_trajectory.copy()
        })

if len(saved_trajectories) > 10:
    saved_trajectories = saved_trajectories[-10:]

# Genera e salva i plot delle traiettorie salvate.
for traj in saved_trajectories:
    plt.figure(figsize=(7, 7))
    puck_traj = np.array(traj["puck"])
    agent_traj = np.array(traj["agent"])
    plt.plot(puck_traj[:, 1], puck_traj[:, 0], 'bo-', label="Puck trajectory")
    plt.plot(agent_traj[:, 1], agent_traj[:, 0], 'ro-', label="Agent trajectory")
    plt.title(f"Trajectory - Episode {traj['episode']}")
    plt.xlabel("Colonna")
    plt.ylabel("Riga")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.savefig(f"trajectory_episode_{traj['episode']}.png")
    plt.close()

# Stampa il reward medio ogni mille episodi
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
count = 1000
print("******** Average reward per thousand episodes ********\n")
for r in rewards_per_thousand_episodes:
    print(f"{count}: {sum(r) / 1000}")
    count += 1000

print("\n\n******** Q-table ********\n")
print(q_table)
print("Top Q-values:", np.max(q_table, axis=1)[:10])
# Funzione di visualizzazione colorata delle traiettorie
def plot_colored_trajectory(traj, grid_size=7):
    fig, ax = plt.subplots(figsize=(7, 7))
    
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)
    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xticklabels(np.arange(grid_size))
    ax.set_yticklabels(np.arange(grid_size))
    
    total_actions = len(traj["agent"])
    ax.set_title(f"Episode {traj['episode']} - Totale azioni agente: {total_actions}")
    
    for i, (puck_pos, agent_pos) in enumerate(zip(traj["puck"], traj["agent"])):
        puck_r, puck_c = puck_pos
        agent_r, agent_c = agent_pos
        
        if puck_pos == list(agent_pos):
            rect = patches.Rectangle((agent_c - 0.5, agent_r - 0.5), 1, 1,
                                     facecolor="green", alpha=0.7, edgecolor="black")
            ax.add_patch(rect)
            ax.text(agent_c, agent_r, f"HIT\n({i})", color="white",
                    ha="center", va="center", fontsize=10, weight="bold")
        else:
            rect_puck = patches.Rectangle((puck_c - 0.5, puck_r - 0.5), 1, 1,
                                          facecolor="blue", alpha=0.5, edgecolor="black")
            ax.add_patch(rect_puck)
            ax.text(puck_c, puck_r, f"P{i}", color="white",
                    ha="center", va="center", fontsize=10, weight="bold")
            
            rect_agent = patches.Rectangle((agent_c - 0.5, agent_r - 0.5), 1, 1,
                                           facecolor="red", alpha=0.5, edgecolor="black")
            ax.add_patch(rect_agent)
            ax.text(agent_c, agent_r, f"A{i}", color="white",
                    ha="center", va="center", fontsize=10, weight="bold")
    
    plt.savefig(f"trajectory_episode_{traj['episode']}.png")
    plt.close()

for traj in saved_trajectories:
    plot_colored_trajectory(traj, grid_size=size)

plt.figure(figsize=(10, 5))
plt.plot(np.arange(1000, num_episodes + 1, 1000), success_rates, color='green', linewidth=2)
plt.xlabel('Episodi')
plt.ylabel('Percentuale di Successo (%)')
plt.title("Andamento della Percentuale di Successo durante l'addestramento")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import matplotlib.patches as patches
import matplotlib.animation as animation
import os
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.animation import FFMpegWriter
import pandas as pd

def create_video(traj, video_filename, grid_size=7):
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

    puck_plot, = ax.plot([], [], 'bo', markersize=15)  # puck
    agent_plot, = ax.plot([], [], 'ro', markersize=25)  # agente
    hit_text = ax.text(grid_size/2, grid_size/2, "", fontsize=20, color="black", ha='center', va='center')

    def init():
        puck_plot.set_data([], [])
        agent_plot.set_data([], [])
        hit_text.set_text("")
        return puck_plot, agent_plot, hit_text

    def update(frame):
        puck_pos = traj["puck"][frame]
        agent_pos = traj["agent"][frame]
        
        puck_plot.set_data([puck_pos[1]], [puck_pos[0]])  # puck
        agent_plot.set_data([agent_pos[1]], [agent_pos[0]])  # agente

        if puck_pos == list(agent_pos):
            hit_text.set_text("HIT")
        else:
            hit_text.set_text("")

        return puck_plot, agent_plot, hit_text

    # Crea l'animazione
    ani = FuncAnimation(fig, update, frames=len(traj["puck"]), init_func=init, blit=True)

    # Usa FFMpegWriter per salvare il video .mp4
    writer = FFMpegWriter(fps=30)
    ani.save(video_filename, writer=writer)

    plt.close(fig)





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
        # Il puck parte dalla riga 0 e da una colonna casuale
        self.object_pos = [0, np.random.randint(0, size)]
    
    def step(self):
        if self.object_pos[0] < self.size - 1:
            self.object_pos[0] += 1  # si muove verso il basso
        else:
            self.reset_position()   # se arriva in fondo, riparte dalla cima
    
    def reset_position(self):
        self.object_pos = [0, np.random.randint(0, self.size)]

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
#     if horizontal_distance != 0 and action == 0:
#         return 20
#     # Se l'agente è allineato orizzontalmente, il reward aumenta man mano che il puck si avvicina alla target row.
#     if horizontal_distance == 0:
#         if error > 0:  # il puck non è ancora arrivato alla target row
#             return 50 - 10 * error  # error minore → reward maggiore
#         # elif error == 0:
#         #     # Se il puck è esattamente nella target row (ma non ancora in collisione)
#         #     return 0
#         else:  # il puck ha superato la target row
#             return -100 - 10 * abs(error)
#     else:
#         # Se l'agente non è allineato, penalizziamo in base alla distanza orizzontale.
#         return -50 - 10 * horizontal_distance


def compute_reward(env, agent):
    puck_r, puck_c = env.object_pos
    agent_r, agent_c = agent.position
    # Se c'è HIT:
    if [puck_r, puck_c] == list(agent.position):
        return 100
    
    # Determina la riga target per il colpo in base alla posizione scelta dall'agente
    target_row = get_target_row(agent.position)
    error = target_row - puck_r  # quanti step mancano al raggiungimento della target row
    column_distance = abs(agent_c - puck_c)
    if column_distance != 0 and action == 0:
        return 25
    # Se l'agente è allineato orizzontalmente, il reward aumenta man mano che il puck si avvicina alla target row.
    if column_distance == 0:
        if error > 0:  # il puck non è ancora arrivato alla target row
            return 50 - 10 * error  # error minore → reward maggiore
        # elif error == 0:
        #     # Se il puck è esattamente nella target row (ma non ancora in collisione)
        #     return 0
        else:  # il puck ha superato la target row
            return -100 - 10 * abs(error)
    else:
        # Se l'agente non è allineato, penalizziamo in base alla distanza orizzontale.
        return -50 - 10 * column_distance






# Parametri di simulazione e Q-learning
size = 7                   # griglia 7x7
num_episodes = 300000       # numero totale di episodi
max_steps_per_episode = size  # numero massimo di step per episodio

learning_rate = 0.1
discount_rate = 0.995

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.001
exploration_decay_rate = 0.001

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

def update(frame, env, agent, action, done):
    if frame > 0:
        env.step()
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
q_table = np.zeros((state_space_size, action_space_size))
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
    
    for step in range(max_steps_per_episode):
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state_index, :])
        else:
            action = random.randint(0, action_space_size - 1)
        
        new_state, reward, done = update(step, env, agent, action, done)
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
    if episode > (num_episodes * 0.9):
        # print(f"Episode {episode}: Reward = {rewards_current_episode}")
        # print("Puck position:", env.object_pos)
        # print("Agent position:", agent.position)
        saved_trajectories.append({
            "episode": episode,
            "puck": puck_trajectory.copy(),
            "agent": agent_trajectory.copy()
        })

if len(saved_trajectories) > 10:
    saved_trajectories = saved_trajectories[-10:]


# Se q_table è un DataFrame di Pandas
with open("q_table.txt", "w") as f:
    f.write(str(q_table))

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

    
for traj in saved_trajectories:
    video_filename = f"trajectory_episode_{traj['episode']}.mp4"
    create_video(traj, video_filename)


plt.figure(figsize=(10, 5))
plt.plot(np.arange(1000, num_episodes + 1, 1000), success_rates, color='green', linewidth=2)
plt.xlabel('Episodes')
plt.ylabel('Success (%)')
plt.title("Success rate during training")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
import pandas as pd

# --------------------------
# Funzione per creare un video a partire da una traiettoria
# --------------------------
def create_video(traj, video_filename, grid_size=21):
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

    # Calcola la dimensione proporzionale del marker
    marker_size = (7 / grid_size) * 9  # Scala rispetto alla figura (7x7 pollici)

    puck_plot, = ax.plot([], [], 'bo', markersize=5)   # disegna il puck
    agent_plot, = ax.plot([], [], 'ro', markersize=5)  # disegna l'agente
    hit_text = ax.text(grid_size/2, grid_size/2, "", fontsize=20, color="black", ha='center', va='center')


    def init():
        puck_plot.set_data([], [])
        agent_plot.set_data([], [])
        hit_text.set_text("")
        return puck_plot, agent_plot, hit_text

    def update(frame):
        puck_pos = traj["puck"][frame]
        agent_pos = traj["agent"][frame]
        puck_plot.set_data([puck_pos[1]], [puck_pos[0]])
        agent_plot.set_data([agent_pos[1]], [agent_pos[0]])
        if puck_pos == list(agent_pos):
            hit_text.set_text("HIT")
        else:
            hit_text.set_text("")
        return puck_plot, agent_plot, hit_text

    ani = FuncAnimation(fig, update, frames=len(traj["puck"]), init_func=init, blit=True)
    writer = FFMpegWriter(fps=30)
    ani.save(video_filename, writer=writer)
    plt.close(fig)

# --------------------------
# Classe dell'ambiente: il puck si muove verso il basso
# --------------------------
class MovingObject:
    def __init__(self, size=21):
        self.size = size
        # Genera la posizione iniziale: riga 0 e colonna casuale
        self.object_pos = [0, np.random.randint(0, size)]
        self.t = 0  # Contatore degli step

    def step(self):
        self.t += 1  # Incrementa ad ogni step
        if self.object_pos[0] < self.size - 1:
            self.object_pos[0] += 1
        else:
            self.reset_position()

    def reset_position(self):
        # In caso di mancato HIT, il puck viene riposizionato nella stessa colonna
        col = self.object_pos[1]  # Mantiene la stessa colonna
        self.object_pos = [0, col]
        self.t = 0

# --------------------------
# Classe dell'agente: movimento graduale con 3 azioni
# --------------------------
class Agent:
    WAITING = 0
    GOING_RIGHT = 1
    GOING_LEFT = 2
    # GO_STRAIGHT = 3
    
    def __init__(self, size=21):
        self.size = size
        self.position = (20, 10)  # Posizione iniziale centrale in basso
        
        num_steps = 10  # 10 sotto-passi per traiettoria
        
        # Generate left trajectory: (20,10) → (10,0)
        self.path_left = []
        for i in range(11):  # 11 positions gives 10 moves
            row = 20 - i
            col = 10 - i
            self.path_left.append((row, col))
        
        # Generate right trajectory: (20,10) → (10,20)
        self.path_right = []
        for i in range(11):
            row = 20 - i
            col = 10 + i
            self.path_right.append((row, col))

        self.fsm_state = Agent.WAITING
        self.target_path = []
        self.action_steps_remaining = 0

    def move(self, action):
        """
        Azioni possibili:
          0 = STAY
          1 = GOTO_RIGHT
          2 = GOTO_LEFT
        Una volta avviato un path, si completa senza poterlo interrompere.
        """
        
        # Se l'agente sta ancora percorrendo un path (a sinistra o destra),
        # deve finire tutti i passi senza cambiare azione.
        if self.action_steps_remaining > 0:
            # Continua il movimento
            self.position = self.target_path.pop(0)
            self.action_steps_remaining -= 1
            
            # Se abbiamo finito i passi, ritorniamo in WAITING
            if self.action_steps_remaining == 0:
                self.fsm_state = Agent.WAITING
            return
        
        # Se invece siamo in WAITING, interpretiamo l'azione scelta
        if action == 0:
            # STAY: resta fermo in waiting
            self.fsm_state = Agent.WAITING
        elif action == 1:
            # Avvia path verso destra
            self.fsm_state = Agent.GOING_RIGHT
            self.target_path = self.path_right.copy()
            self.action_steps_remaining = len(self.target_path)
            self.position = self.target_path.pop(0)
            self.action_steps_remaining -= 1
        elif action == 2:
            # Avvia path verso sinistra
            self.fsm_state = Agent.GOING_LEFT
            self.target_path = self.path_left.copy()
            self.action_steps_remaining = len(self.target_path)
            self.position = self.target_path.pop(0)
            self.action_steps_remaining -= 1

# --------------------------
# State space
# --------------------------
def state_to_index(puck_state, agent_state, grid_size, num_agent_positions):
    # Calcola un indice univoco combinando la posizione del puck e quella dell'agente
    puck_index = puck_state[0] * grid_size + puck_state[1]
    # return puck_index * num_agent_positions + agent_state
    return puck_index * num_agent_positions

# def get_state(env, agent_fsm, grid_size):
#     # Puck index (come prima)
#     puck_r, puck_c = env.object_pos
#     puck_idx = puck_r * grid_size + puck_c
    
#     # Agent_fsm è 0/1/2
#     # Lo “stato globale” è (puck_idx, agent_fsm) fuso in un solo indice
#     return puck_idx * 3 
#     # return puck_idx * 3 + agent_fsm

def get_state(env, grid_size):
    puck_r, puck_c = env.object_pos
    return puck_r * grid_size + puck_c


def compute_reward(env, agent):
    
    if env.object_pos == list(agent.position):
        return 1500 
    
    if agent.fsm_state == Agent.WAITING:
        return 0

    
    return -10


size = 21
num_episodes = int(10e5)
max_steps_per_episode = size  # ad es., 7 step per episodio

learning_rate = 0.1
discount_rate = 1

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.001
exploration_decay_rate = 0.0005

n_runs = 15
cmap = plt.get_cmap('Spectral')

colors = [(*cmap(i/(n_runs - 1))[:3], 1) for i in range(n_runs)]
plt.figure(figsize=(20, 10))
plt.xlabel('Episodes')
plt.ylabel('Success (%)')
plt.title(f"Success rate during training over {n_runs} runs")
plt.grid(True, linestyle='--', alpha=0.6)

all_success_rates = []

# --------------------------
# Q learning cycle
# --------------------------
for run in range(n_runs):
    print(f"Run {run+1}")
    env = MovingObject(size)
    agent = Agent(size)
    action_space_size = 3 
    state_space_size = (size * size) #*3
    q_table = np.zeros((state_space_size, action_space_size))
    successes = []      # per registrare se ogni episodio ha portato a HIT (1) o meno (0)
    success_rates = []  # per salvare la percentuale di successo ogni 1000 episodi
    saved_trajectories = []  # per salvare alcune traiettorie (per visualizzazione)
    sample_interval = num_episodes // 1000
    episodes_data = []

    for episode in tqdm(range(num_episodes), desc="Training episodes"):
        env = MovingObject(size)
        agent = Agent(size)  # reset dell'agente
        # state_index = get_state(env, agent.fsm_state, size)
        state_index = get_state(env, size)
        done = False
        episode_success = 0
        puck_trajectory = []
        agent_trajectory = []
        
        
        # Inizializza la variabile che terrà traccia dell'azione in corso
        current_action = None
        
        for step in range(max_steps_per_episode):
            #i want to save the first position of the agent and the puck in puck_trajectory and agent_trajectory
            if step == 0:
                puck_trajectory.append(env.object_pos.copy())
                agent_trajectory.append(list(agent.position))

            # Epsilon-greedy selection
            if random.uniform(0, 1) > exploration_rate:
                action = np.argmax(q_table[state_index, :])
            else:
                action = random.randint(0, action_space_size - 1)
            
            # Aggiorna l'ambiente
            env.step()
            # Muovi l'agente in base alla sua FSM
            agent.move(action)
            
            # Calcola nuovo stato
            # new_state_index = get_state(env, agent.fsm_state, size)
            new_state_index = get_state(env, size)

            # Calcola reward (stesse logiche di prima, ma usando agent.position per la collisione)
            reward = compute_reward(env, agent)

            # Q-update
            q_table[state_index, action] = q_table[state_index, action] * (1 - learning_rate) + \
                learning_rate * (reward + discount_rate * np.max(q_table[new_state_index, :]))

            state_index = new_state_index
            if(num_episodes-episode)<=1000:
                puck_trajectory.append(env.object_pos.copy())
                agent_trajectory.append(list(agent.position))


            
            # Se c'è HIT, termina l'episodio
            if env.object_pos == list(agent.position):
                episode_success = 1
                break
            
        if(num_episodes-episode)<=1000:
            episodes_data.append({
                "episode": episode,
                "puck": puck_trajectory,
                "agent": agent_trajectory
                }) 

        
        successes.append(episode_success)
        # Update of the exploration rate, exponential decay
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
        
        # if (episode + 1) % 1000 == 0 :
        if (episode + 1) % sample_interval == 0:
            recent_success_rate = sum(successes[-sample_interval:]) / sample_interval * 100
            success_rates.append(recent_success_rate)
            print("\n",recent_success_rate)
        # Salva le traiettorie degli ultimi episodi (ultimi 10%) se c'è stato HIT
        if(num_episodes-episode)<=1000 and episode_success == 1:
            saved_trajectories.append({
                "episode": episode,
                "puck": puck_trajectory.copy(),
                "agent": agent_trajectory.copy()
            })
    
    all_success_rates.append(success_rates)
    if len(saved_trajectories) > 10:
        saved_trajectories = saved_trajectories[-10:]
    
    # Salvataggio opzionale della Q-table su file
    # with open("q_table.txt", "w") as f:
    #     f.write(str(q_table))
    
    # --------------------------
    # grid plot
    # --------------------------
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
        ax.set_title(f"Episode {traj['episode']}")
        
        for i, (puck_pos, agent_pos) in enumerate(zip(traj["puck"], traj["agent"])):
            if puck_pos == list(agent_pos):
                rect = patches.Rectangle((agent_pos[1]-0.5, agent_pos[0]-0.5), 1, 1,
                                         facecolor="green", alpha=0.7, edgecolor="black")
                ax.add_patch(rect)
                ax.text(agent_pos[1], agent_pos[0], f"HIT\n({i})", color="white",
                        ha="center", va="center", fontsize=10, weight="bold")
            else:
                rect_puck = patches.Rectangle((puck_pos[1]-0.5, puck_pos[0]-0.5), 1, 1,
                                              facecolor="blue", alpha=0.5, edgecolor="black")
                ax.add_patch(rect_puck)
                ax.text(puck_pos[1], puck_pos[0], f"P{i}", color="white",
                        ha="center", va="center", fontsize=10, weight="bold")
                rect_agent = patches.Rectangle((agent_pos[1]-0.5, agent_pos[0]-0.5), 1, 1,
                                               facecolor="red", alpha=0.5, edgecolor="black")
                ax.add_patch(rect_agent)
                ax.text(agent_pos[1], agent_pos[0], f"A{i}", color="white",
                        ha="center", va="center", fontsize=10, weight="bold")
        
        plt.close()

    for traj in saved_trajectories:
        

        plot_colored_trajectory(traj, grid_size=size)
        video_filename = f"trajectory_episode_{traj['episode']}.mp4"
        create_video(traj, video_filename)

    sample_interval = num_episodes // 1000
    plt.plot(np.arange(sample_interval, num_episodes + 1, sample_interval), success_rates, color=colors[run], linewidth=2, label=f'Run {run+1}')
    plt.savefig("success_rate_during_runs.png")
    plt.pause(0.5)

# --------------------------
#plots
# --------------------------
# Ultimi 10 episodi
last_10_episodes = episodes_data[-10:] if len(episodes_data) > 10 else episodes_data

for ep_data in last_10_episodes:
    # 1) Creiamo la matrice di occupazione dell'AGENTE
    occupant_matrix = np.zeros((size, size), dtype=int)
    for (r, c) in ep_data["agent"]:
        occupant_matrix[r, c] += 1

    # 2) Costruiamo la figure e la heatmap
    fig, ax = plt.subplots(figsize=(5, 4))
    heatmap = ax.imshow(occupant_matrix, cmap='viridis', origin='upper')
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label("Times visited by Agent")

    # 3) Tracciamo i cerchi del puck
    #    ep_data["puck"] = lista di posizioni (r, c) in cui si trova il puck a ogni step
    #    scatter: x = col, y = row
    puck_positions = ep_data["puck"]
    puck_cols = [pos[1] for pos in puck_positions]
    puck_rows = [pos[0] for pos in puck_positions]

    # disegniamo i cerchi con marker='o', un colore a piacere (es. rosso), dimensione (s=80)
    ax.scatter(puck_cols, puck_rows, marker='o', s=80, 
               facecolors='none', edgecolors='r', linewidths=1.5,
               label="Puck path")

    ax.set_title(f"Heatmap - Episode {ep_data['episode']}")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    ax.legend(loc="upper right")

    # 4) Salviamo
    plt.savefig(f"heatmap_episode_{ep_data['episode']}.png")
    plt.close(fig)

all_success_rates = np.array(all_success_rates)
episodes_axis = np.arange(1000, num_episodes + 1, 1000)
mean_sr = np.mean(all_success_rates, axis=0)
std_sr = np.std(all_success_rates, axis=0)
lower_bound = mean_sr - std_sr
upper_bound = mean_sr + std_sr
final_mean = mean_sr[-1]
first_quarter_mean = np.mean(mean_sr[:len(mean_sr)//4])
mid_training_mean = np.mean(mean_sr[len(mean_sr)//2:len(mean_sr)//2 + len(mean_sr)//4])
final_std = std_sr[-1]

plt.figure(figsize=(10, 6))
plt.plot(episodes_axis, mean_sr, color="darkslateblue", linewidth=2, label='Mean success rate')
plt.fill_between(episodes_axis, lower_bound, upper_bound, color="darkslateblue", alpha=0.3, label='± 1 Std Dev')
plt.xlabel('Episodes')
plt.ylabel('Success (%)')
plt.title('Aggregated Success Rate with Variance')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

textstr = f"Final Mean: {final_mean:.2f}%\nFinal Std Dev: {final_std:.2f}%\nMean (1st Quarter): {first_quarter_mean:.2f}%\nMean (Mid Training): {mid_training_mean:.2f}%"
plt.gca().text(0.02, 0.02, textstr, transform=plt.gca().transAxes,
               fontsize=12, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))
plt.savefig("aggregated_success_rate_cold_purple.png")
plt.show()



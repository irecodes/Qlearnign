import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
import pandas as pd

# --------------------------
# Funzione per creare un video a partire da una traiettoria
# --------------------------
def create_video(traj, video_filename, grid_rows, grid_cols):
    fig, ax = plt.subplots(figsize=(7, 7))
    # Imposta i tick e la griglia per le colonne (asse x)
    ax.set_xticks(np.arange(-0.5, grid_cols, 1), minor=True)
    # Imposta i tick e la griglia per le righe (asse y)
    ax.set_yticks(np.arange(-0.5, grid_rows, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.set_xlim(-0.5, grid_cols - 0.5)
    ax.set_ylim(grid_rows - 0.5, -0.5)
    ax.set_xticks(np.arange(grid_cols))
    ax.set_yticks(np.arange(grid_rows))
    ax.set_xticklabels(np.arange(grid_cols))
    ax.set_yticklabels(np.arange(grid_rows))

    # Calcola la dimensione proporzionale del marker (opzionale)
    marker_size = (7 / grid_cols) * 9

    puck_plot, = ax.plot([], [], 'bo', markersize=5)   # disegna il puck
    agent_plot, = ax.plot([], [], 'ro', markersize=5)    # disegna l'agente
    hit_text = ax.text(grid_cols/2, grid_rows/2, "", fontsize=20,
                       color="black", ha='center', va='center')

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

    ani = FuncAnimation(fig, update, frames=len(traj["puck"]),
                        init_func=init, blit=True)
    writer = FFMpegWriter(fps=30)
    ani.save(video_filename, writer=writer)
    plt.close(fig)

# --------------------------
# Classe dell'ambiente: il puck si muove verso il basso
# --------------------------
class MovingObject:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        # Il puck parte sempre dalla riga 0 e dalla colonna 0 (o si può randomizzare la colonna)
        self.object_pos = [0, np.random.randint(0, cols-1)]
        self.t = 0  # Contatore degli step

    def step(self):
        self.t += 1  # Incrementa ad ogni step
        if self.object_pos[0] < self.rows - 1:
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
    
    def __init__(self, grid_rows, grid_cols):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        # Posizione iniziale traslata: ora l'agente parte dalla riga inferiore (22-esima, indice 21)
        self.position = (grid_rows - 1, grid_cols // 2)
        
        num_steps = 10  # 10 sotto-passi per traiettoria
        
        # Genera traiettoria verso sinistra: da (grid_rows-1, grid_cols//2) a (grid_rows-1 - num_steps, grid_cols//2 - num_steps)
        self.path_left = []
        for i in range(num_steps + 1):
            row = (grid_rows - 1) - i
            col = (grid_cols // 2) - i
            self.path_left.append((row, col))
        
        # Genera traiettoria verso destra: da (grid_rows-1, grid_cols//2) a (grid_rows-1 - num_steps, grid_cols//2 + num_steps)
        self.path_right = []
        for i in range(num_steps + 1):
            row = (grid_rows - 1) - i
            col = (grid_cols // 2) + i
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
        # Se l'agente sta ancora percorrendo un path, continua il movimento
        if self.action_steps_remaining > 0:
            self.position = self.target_path.pop(0)
            self.action_steps_remaining -= 1
            if self.action_steps_remaining == 0:
                self.fsm_state = Agent.WAITING
            return
        
        # Se siamo in WAITING, interpretiamo l'azione scelta
        if action == 0:
            self.fsm_state = Agent.WAITING
        elif action == 1:
            self.fsm_state = Agent.GOING_RIGHT
            self.target_path = self.path_right.copy()
            self.action_steps_remaining = len(self.target_path)
            self.position = self.target_path.pop(0)
            self.action_steps_remaining -= 1
        elif action == 2:
            self.fsm_state = Agent.GOING_LEFT
            self.target_path = self.path_left.copy()
            self.action_steps_remaining = len(self.target_path)
            self.position = self.target_path.pop(0)
            self.action_steps_remaining -= 1

# --------------------------
# Funzioni per lo state space e il reward
# --------------------------
def get_state(env, grid_cols):
    # Stato rappresentato dalla posizione del puck
    puck_r, puck_c = env.object_pos
    return puck_r * grid_cols + puck_c

def compute_reward(env, agent):
    if env.object_pos == list(agent.position):
        return 1500 
    if agent.fsm_state == Agent.WAITING:
        return 0.5
    return -15

# --------------------------
# Parametri dell'environment e training
# --------------------------
grid_rows = 23  # Nuova dimensione: 22 righe (0...21)
grid_cols = 21  # 21 colonne

num_episodes = int(14e5)
max_steps_per_episode = grid_rows  # ad es., 22 step per episodio

learning_rate = 0.1
discount_rate = 1

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.0005
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
# Q-learning cycle
# --------------------------
for run in range(n_runs):
    print(f"Run {run+1}")
    env = MovingObject(grid_rows, grid_cols)
    agent = Agent(grid_rows, grid_cols)
    action_space_size = 3 
    state_space_size = grid_rows * grid_cols
    q_table = np.zeros((state_space_size, action_space_size))
    successes = []      # registra se ogni episodio ha portato a HIT (1) o meno (0)
    success_rates = []  # per salvare la percentuale di successo ogni 1000 episodi
    saved_trajectories = []  # per salvare alcune traiettorie (per visualizzazione)
    sample_interval = num_episodes // 1000
    episodes_data = []

    for episode in tqdm(range(num_episodes), desc="Training episodes"):
        env = MovingObject(grid_rows, grid_cols)
        agent = Agent(grid_rows, grid_cols)  # reset dell'agente
        state_index = get_state(env, grid_cols)
        episode_success = 0
        puck_trajectory = []
        agent_trajectory = []
        
        for step in range(max_steps_per_episode):
            if step == 0:
                puck_trajectory.append(env.object_pos.copy())
                agent_trajectory.append(list(agent.position))

            # Selezione epsilon-greedy
            if random.uniform(0, 1) > exploration_rate:
                action = np.argmax(q_table[state_index, :])
            else:
                action = random.randint(0, action_space_size - 1)
            
            # Aggiorna ambiente e agente
            env.step()
            agent.move(action)
            new_state_index = get_state(env, grid_cols)
            reward = compute_reward(env, agent)

            # Aggiornamento Q-table
            q_table[state_index, action] = q_table[state_index, action] * (1 - learning_rate) + \
                learning_rate * (reward + discount_rate * np.max(q_table[new_state_index, :]))
            state_index = new_state_index

            if (num_episodes - episode) <= 1000:
                puck_trajectory.append(env.object_pos.copy())
                agent_trajectory.append(list(agent.position))
            
            # Se c'è HIT, termina l'episodio
            if env.object_pos == list(agent.position):
                episode_success = 1
                break
            
        if (num_episodes - episode) <= 1000:
            episodes_data.append({
                "episode": episode,
                "puck": puck_trajectory,
                "agent": agent_trajectory
            })
        
        successes.append(episode_success)
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
        
        if (episode + 1) % sample_interval == 0:
            recent_success_rate = sum(successes[-sample_interval:]) / sample_interval * 100
            success_rates.append(recent_success_rate)
            print("\n", recent_success_rate)
        if (num_episodes - episode) <= 1000 and episode_success == 1:
            saved_trajectories.append({
                "episode": episode,
                "puck": puck_trajectory.copy(),
                "agent": agent_trajectory.copy()
            })
    
    all_success_rates.append(success_rates)
    if len(saved_trajectories) > 10:
        saved_trajectories = saved_trajectories[-10:]
    
    # --------------------------
    # Plot della traiettoria (video e grid plot)
    # --------------------------
    def plot_colored_trajectory(traj, grid_rows, grid_cols):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xticks(np.arange(-0.5, grid_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_rows, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        ax.set_xlim(-0.5, grid_cols - 0.5)
        ax.set_ylim(grid_rows - 0.5, -0.5)
        ax.set_xticks(np.arange(grid_cols))
        ax.set_yticks(np.arange(grid_rows))
        ax.set_xticklabels(np.arange(grid_cols))
        ax.set_yticklabels(np.arange(grid_rows))
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
        plot_colored_trajectory(traj, grid_rows, grid_cols)
        video_filename = f"trajectory_episode_{traj['episode']}.mp4"
        create_video(traj, video_filename, grid_rows, grid_cols)
    
    episodes_axis = np.arange(sample_interval, num_episodes + 1, sample_interval)
    plt.plot(episodes_axis, success_rates, color=colors[run], linewidth=2, label=f'Run {run+1}')
    plt.savefig("success_rate_during_runs.png")
    plt.pause(0.5)

# --------------------------
# Plot finali: Heatmaps e Success Rate aggregata
# --------------------------
# Heatmaps per gli ultimi 10 episodi
last_10_episodes = episodes_data[-10:] if len(episodes_data) > 10 else episodes_data

for ep_data in last_10_episodes:
    occupant_matrix = np.zeros((grid_rows, grid_cols), dtype=int)
    for (r, c) in ep_data["agent"]:
        occupant_matrix[r, c] += 1

    fig, ax = plt.subplots(figsize=(5, 4))
    heatmap = ax.imshow(occupant_matrix, cmap='viridis', origin='upper')
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label("Times visited by Agent")

    puck_positions = ep_data["puck"]
    puck_cols = [pos[1] for pos in puck_positions]
    puck_rows = [pos[0] for pos in puck_positions]

    ax.scatter(puck_cols, puck_rows, marker='o', s=80, 
               facecolors='none', edgecolors='r', linewidths=1.5,
               label="Puck path")

    ax.set_title(f"Heatmap - Episode {ep_data['episode']}")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.legend(loc="upper right")

    plt.savefig(f"heatmap_episode_{ep_data['episode']}.png")
    plt.close(fig)

all_success_rates = np.array(all_success_rates)
episodes_axis = np.arange(sample_interval, num_episodes + 1, sample_interval)
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

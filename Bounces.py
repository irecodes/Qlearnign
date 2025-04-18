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
    ax.set_xticks(np.arange(-0.5, grid_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_rows, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.set_xlim(-0.5, grid_cols - 0.5)
    ax.set_ylim(grid_rows - 0.5, -0.5)
    ax.set_xticks(np.arange(grid_cols))
    ax.set_yticks(np.arange(grid_rows))
    ax.set_xticklabels(np.arange(grid_cols))
    ax.set_yticklabels(np.arange(grid_rows))

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
# Classe dell'ambiente: il puck si muove verso il basso e in diagonale
# --------------------------
class MovingObject:
    def __init__(self, rows, cols, phase=1):
        self.rows = rows
        self.cols = cols
        self.t = 0
        self.phase = phase 
        self.set_dx()
        if self.dx == 0:
            self.object_pos = [0, np.random.randint(0, cols)]
        elif self.dx == 1:
            self.object_pos = [0, np.random.choice([0,2,4,6,8, 10, 12, 14, 16, 18])]
        else:
            self.object_pos = [0, np.random.choice([20, 18, 16,14,12, 2, 4, 6, 8, 10])]

    def set_dx(self):
        if self.phase == 1:
            self.dx = 0
        elif self.phase == 2:
            self.dx = -1
        elif self.phase == 3:
            self.dx = np.random.choice([-1, 0, 1])
            
    def step(self):
        self.t += 1
        new_row = self.object_pos[0] + 1 
        new_col = self.object_pos[1] + self.dx  
        # Gestione del rimbalzo orizzontale: se si esce dai limiti, inverte la direzione
        if new_col < 0 or new_col >= self.cols:
            if self.dx != 0:
                self.dx = -self.dx  # inverte la direzione orizzontale
                new_col = self.object_pos[1] + self.dx  # ricalcola la colonna
        self.object_pos[0] = new_row
        self.object_pos[1] = new_col
        # Reset se il puck ha raggiunto la parte inferiore
        if self.object_pos[0] >= self.rows-1:
            self.reset_position()

    def reset_position(self):
        self.object_pos[0] = 0
        # La colonna rimane invariata
        self.t = 0
        self.set_dx()

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
        # L'agente parte dalla riga inferiore, centro della griglia
        self.position = (grid_rows - 1, grid_cols // 2)
        num_steps = 10  # 10 sotto-passi per traiettoria
        # Traiettoria verso sinistra
        self.path_left = []
        for i in range(num_steps + 1):
            row = (grid_rows - 1) - i
            col = (grid_cols // 2) - i
            self.path_left.append((row, col))
        # Traiettoria verso destra
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
        Una volta avviato un path, lo completa senza poterlo interrompere.
        """
        if self.action_steps_remaining > 0:
            self.position = self.target_path.pop(0)
            self.action_steps_remaining -= 1
            if self.action_steps_remaining == 0:
                self.fsm_state = Agent.WAITING
            return
        
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
    """
    Codifica lo stato come un indice univoco che combina:
    - la riga del puck
    - la colonna del puck
    - la direzione orizzontale del puck: -1 -> 0, 0 -> 1, +1 -> 2
    """
    puck_r, puck_c = env.object_pos
    if env.dx == -1:
        direction = 0
    elif env.dx == 0:
        direction = 1
    else:  # env.dx == +1
        direction = 2
    return (puck_r * grid_cols + puck_c) * 3 + direction

def compute_reward(env, agent, progress):
    # Se l'agente raggiunge l'obiettivo, reward elevato
    if env.object_pos == list(agent.position):
        return 2000 
    # Se l'agente sta aspettando, reward positivo moderato
    if agent.fsm_state == Agent.WAITING:
        return 80
    # Penalità per movimenti non motivati che aumenta con il progresso:
    # All'inizio la penalità è leggera (ad esempio -10) e diventa più severa
    penalty = -10 * (1-progress) - 25 * progress  
    return penalty

# --------------------------
# Parametri dell'environment e training
# --------------------------
grid_rows = 23
grid_cols = 21

num_episodes = int(35e5)
max_steps_per_episode = 22

# Parametri per il Q-learning con eligibility traces
learning_rate = 0.05
discount_rate = 0.95 # leggermente inferiore ad 1 per favorire la stabilità
lambda_trace = 0.85    # parametro lambda per le eligibility traces

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.0005
exploration_decay_rate = 0.000005

n_runs = 5
cmap = plt.get_cmap('Spectral')
colors = [(*cmap(i/(n_runs - 1))[:3], 1) for i in range(n_runs)]

plt.figure(figsize=(20, 10))
plt.xlabel('Episodes')
plt.ylabel('Success (%)')
plt.title(f"Success rate during training over {n_runs} runs")
plt.grid(True, linestyle='--', alpha=0.6)

all_success_rates = []

# --------------------------
# Ciclo di Q-learning con Eligibility Traces
# --------------------------
for run in range(n_runs):
    print(f"Run {run+1}")
    phase = 3  # inizialmente puoi settare la fase che preferisci

    state_space_size = grid_rows * grid_cols * 3
    action_space_size = 3 
    q_table = np.zeros((state_space_size, action_space_size))
    
    successes = []
    success_rates = []
    saved_trajectories = []
    sample_interval = num_episodes // 1000
    episodes_data = []

    for episode in tqdm(range(num_episodes), desc="Training episodes"):
        # Calcola il progresso corrente: 0 all'inizio, 1 alla fine
        progress = episode / num_episodes
        
        # Inizializza il nuovo ambiente e l'agente per l'episodio corrente
        env = MovingObject(grid_rows, grid_cols, phase=phase)
        agent = Agent(grid_rows, grid_cols)
        
        # Inizializza eligibility traces: stessa dimensione della Q_table
        eligibility_traces = np.zeros_like(q_table)
        
        state_index = get_state(env, grid_cols)
        episode_success = 0
        puck_trajectory = []
        agent_trajectory = []
        
        for step in range(max_steps_per_episode):
            # Salva le traiettorie per visualizzazione (opzionale)
            if step == 0:
                puck_trajectory.append(env.object_pos.copy())
                agent_trajectory.append(list(agent.position))
            
            # Selezione epsilon-greedy
            if random.uniform(0, 1) > exploration_rate:
                action = np.argmax(q_table[state_index, :])
            else:
                action = random.randint(0, action_space_size - 1)
            
            env.step()
            agent.move(action)
            new_state_index = get_state(env, grid_cols)
            reward = compute_reward(env, agent, progress)
            
            # Calcolo del TD error: usa max Q per il nuovo stato
            td_error = reward + discount_rate * np.max(q_table[new_state_index, :]) - q_table[state_index, action]
            
            # Aggiorna le eligibility traces (accumulate traces)
            eligibility_traces[state_index, action] += 1
            
            # Aggiorna Q_table e decay delle eligibility traces per tutti gli stati e azioni
            q_table += learning_rate * td_error * eligibility_traces
            eligibility_traces *= discount_rate * lambda_trace
            
            state_index = new_state_index
            
            # Salvataggio delle traiettorie negli ultimi episodi
            if (num_episodes - episode) <= 1000:
                puck_trajectory.append(env.object_pos.copy())
                agent_trajectory.append(list(agent.position))
            
            # Se l'agente ha centrato il puck, termina l'episodio
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
        # Aggiornamento decaying dell'epsilon
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
        
        if (episode + 1) % sample_interval == 0:
            if len(successes) >= sample_interval:
                recent_success_rate = sum(successes[-sample_interval:]) / sample_interval * 100
                success_rates.append(recent_success_rate)
                if (episode + 1) % (sample_interval * 50) == 0:
                    print("\nSuccess rate:", recent_success_rate)
        
        if (num_episodes - episode) <= 1000:
            saved_trajectories.append({
                "episode": episode,
                "puck": puck_trajectory.copy(),
                "agent": agent_trajectory.copy()
            })
    
    all_success_rates.append(success_rates)
    if len(saved_trajectories) > 20:
        saved_trajectories = saved_trajectories[-20:]
    
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
                        ha='center', va='center', fontsize=10, weight="bold")
            else:
                rect_puck = patches.Rectangle((puck_pos[1]-0.5, puck_pos[0]-0.5), 1, 1,
                                              facecolor="blue", alpha=0.5, edgecolor="black")
                ax.add_patch(rect_puck)
                ax.text(puck_pos[1], puck_pos[0], f"P{i}", color="white",
                        ha='center', va='center', fontsize=10, weight="bold")
                rect_agent = patches.Rectangle((agent_pos[1]-0.5, agent_pos[0]-0.5), 1, 1,
                                               facecolor="red", alpha=0.5, edgecolor="black")
                ax.add_patch(rect_agent)
                ax.text(agent_pos[1], agent_pos[0], f"A{i}", color="white",
                        ha='center', va='center', fontsize=10, weight="bold")
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

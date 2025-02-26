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

    puck_plot, = ax.plot([], [], 'bo', markersize=15)   # disegna il puck
    agent_plot, = ax.plot([], [], 'ro', markersize=25)    # disegna l'agente
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
    def __init__(self, size=7):
        self.size = size
        # Il puck parte dalla riga 0 in una colonna casuale
        self.object_pos = [0, np.random.randint(0, size)]
    
    def step(self):
        if self.object_pos[0] < self.size - 1:
            self.object_pos[0] += 1  # muove il puck verso il basso
        else:
            self.reset_position()
    
    def reset_position(self):
        self.object_pos = [0, np.random.randint(0, self.size)]

# --------------------------
# Classe dell'agente: movimento graduale con 3 azioni
# --------------------------
class Agent:
    WAITING = 0
    GOING_RIGHT = 1
    GOING_LEFT = 2
    # GO_STRAIGHT = 3
    
    def __init__(self, size=7):
        self.size = size
        # Posizione “fisica” interna
        self.position = (size - 1, 3)
        
        self.path_left = [(size - 2, 2), (size - 3, 1), (size - 4, 0)]
        self.path_right = [(size - 2, 4), (size - 3, 5), (size - 4, 6)]
        # self.path_straight = [(size - 2, 3), (size - 3, 3), (size - 4, 3)]
        
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
        # elif action == 3:   
        #     # Avvia path dritto
        #     self.fsm_state = Agent.GO_STRAIGHT
        #     self.target_path = self.path_straight.copy()
        #     self.action_steps_remaining = len(self.target_path)
        #     self.position = self.target_path.pop(0)
        #     self.action_steps_remaining -= 1

# --------------------------
# State space
# --------------------------
def state_to_index(puck_state, agent_state, grid_size, num_agent_positions):
    # Calcola un indice univoco combinando la posizione del puck e quella dell'agente
    puck_index = puck_state[0] * grid_size + puck_state[1]
    return puck_index * num_agent_positions + agent_state

def get_state(env, agent_fsm, grid_size):
    # Puck index (come prima)
    puck_r, puck_c = env.object_pos
    puck_idx = puck_r * grid_size + puck_c
    
    # Agent_fsm è 0/1/2
    # Lo “stato globale” è (puck_idx, agent_fsm) fuso in un solo indice
    return puck_idx * 3 + agent_fsm


# --------------------------
# Reward Function
# --------------------------
def compute_reward(env, agent):
    if env.object_pos == list(agent.position):
        return 1000 
    if agent.position == (6, 3):
        return 7  # wait
    #negative reward if the object surpasses the agent and the agent becomes unable to reach it
    if env.object_pos[0] > agent.position[0]:
        return -350
    return -0.0001 

# --------------------------
# Q-learning parameters
# --------------------------
size = 7  # grig10lia 7x7
num_episodes = 50000
max_steps_per_episode = size + 5  # ad es., 7 step per episodio

learning_rate = 0.1
discount_rate = 0.95

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.0005

n_runs = 100

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
    state_space_size = (size * size) * 3 
    q_table = np.zeros((state_space_size, action_space_size))
    successes = []      # per registrare se ogni episodio ha portato a HIT (1) o meno (0)
    success_rates = []  # per salvare la percentuale di successo ogni 1000 episodi
    saved_trajectories = []  # per salvare alcune traiettorie (per visualizzazione)

    for episode in tqdm(range(num_episodes), desc="Training episodes"):
        env = MovingObject(size)
        agent = Agent(size)  # reset dell'agente
        state_index = get_state(env, agent.fsm_state, size)
        done = False
        episode_success = 0
        puck_trajectory = []
        agent_trajectory = []
        
        # Inizializza la variabile che terrà traccia dell'azione in corso
        current_action = None
        
        for step in range(max_steps_per_episode):
            
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
            new_state_index = get_state(env, agent.fsm_state, size)

            # Calcola reward (stesse logiche di prima, ma usando agent.position per la collisione)
            reward = compute_reward(env, agent)

            # Q-update
            q_table[state_index, action] = q_table[state_index, action] * (1 - learning_rate) + \
                learning_rate * (reward + discount_rate * np.max(q_table[new_state_index, :]))

            state_index = new_state_index
            puck_trajectory.append(env.object_pos.copy())
            agent_trajectory.append(list(agent.position))
            
            # Se c'è HIT, termina l'episodio
            if env.object_pos == list(agent.position):
                episode_success = 1
                break
        
        successes.append(episode_success)
        # Update of the exploration rate, exponential decay
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
        
        if (episode + 1) % 1000 == 0:
            recent_success_rate = sum(successes[-1000:]) / 1000 * 100
            success_rates.append(recent_success_rate)
        
        # Salva le traiettorie degli ultimi episodi (ultimi 10%) se c'è stato HIT
        if episode > (num_episodes * 0.9) and episode_success == 1:
            saved_trajectories.append({
                "episode": episode,
                "puck": puck_trajectory.copy(),
                "agent": agent_trajectory.copy()
            })
    
    all_success_rates.append(success_rates)
    if len(saved_trajectories) > 10:
        saved_trajectories = saved_trajectories[-10:]
    
    # Salvataggio opzionale della Q-table su file
    with open("q_table.txt", "w") as f:
        f.write(str(q_table))
    
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
    
    plt.plot(np.arange(1000, num_episodes + 1, 1000), success_rates, color=colors[run], linewidth=2, label=f'Run {run+1}')
    plt.pause(0.5)

# --------------------------
#plots
# --------------------------
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

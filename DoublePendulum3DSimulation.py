import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# Physikalische Parameter
g = 9.81
damping = 0.01
L1, L2 = 1.0, 1.0
m1, m2 = 1.0, 1.0

# Simulationsparameter
u_range = 1.0
v_range = 1.0
grid_size = 60  # Reduziert für schnellere Berechnung

# Zeitentwicklungsparameter
steps_min = 50
steps_max = 500
steps_step = 2
dt = 0.02

def simulate_pendulum(u, v, steps=50):
    theta1 = np.pi + u * 0.1
    theta2 = np.pi + v * 0.1
    theta1_dot = 0.0
    theta2_dot = 0.0
    
    def calculate_energy(t1, t2, t1d, t2d):
        try:
            y1 = -L1 * np.cos(t1)
            y2 = y1 - L2 * np.cos(t2)
            potential = m1 * g * y1 + m2 * g * y2
            
            x1d = L1 * np.cos(t1) * t1d
            y1d = L1 * np.sin(t1) * t1d
            x2d = x1d + L2 * np.cos(t2) * t2d
            y2d = y1d + L2 * np.sin(t2) * t2d
            
            kinetic = 0.5 * m1 * (x1d**2 + y1d**2) + 0.5 * m2 * (x2d**2 + y2d**2)
            return potential + kinetic
        except:
            return 0.0
    
    initial_energy = 0
    final_energy = 0
    
    for i in range(steps):
        try:
            delta = theta2 - theta1
            sin_delta = np.sin(delta)
            cos_delta = np.cos(delta)
            
            theta1_dot = np.clip(theta1_dot, -10, 10)
            theta2_dot = np.clip(theta2_dot, -10, 10)
            
            denom1 = (m1 + m2) * L1 - m2 * L1 * cos_delta**2
            denom2 = (L2 / L1) * denom1
            
            if abs(denom1) < 1e-10 or abs(denom2) < 1e-10:
                break
                
            theta1_dot_dot = (m2 * L2 * theta2_dot**2 * sin_delta * cos_delta +
                             m2 * g * np.sin(theta2) * cos_delta +
                             m2 * L2 * theta2_dot**2 * sin_delta -
                             (m1 + m2) * g * np.sin(theta1)) / denom1
            
            theta2_dot_dot = (-m2 * L2 * theta2_dot**2 * sin_delta * cos_delta +
                             (m1 + m2) * (g * np.sin(theta1) * cos_delta - g * np.sin(theta2)) -
                             (m1 + m2) * L1 * theta1_dot**2 * sin_delta) / denom2
            
            theta1_dot_dot -= damping * theta1_dot
            theta2_dot_dot -= damping * theta2_dot
            
            theta1_dot_dot = np.clip(theta1_dot_dot, -50, 50)
            theta2_dot_dot = np.clip(theta2_dot_dot, -50, 50)
            
            theta1_dot += theta1_dot_dot * dt
            theta2_dot += theta2_dot_dot * dt
            theta1 += theta1_dot * dt
            theta2 += theta2_dot * dt
            
            if i == 5:
                initial_energy = calculate_energy(theta1, theta2, theta1_dot, theta2_dot)
            if i == steps - 5:
                final_energy = calculate_energy(theta1, theta2, theta1_dot, theta2_dot)
                
        except:
            return float('inf')
    
    energy_loss = abs(initial_energy - final_energy)
    return energy_loss if not np.isnan(energy_loss) else float('inf')

def generate_animation_landscapes():
    u_values = np.linspace(-u_range, u_range, grid_size)
    v_values = np.linspace(-v_range, v_range, grid_size)
    U, V = np.meshgrid(u_values, v_values)
    
    steps_range = range(steps_min, steps_max + 1, steps_step)
    landscapes = []
    
    print("Berechne Landschaften für Zeitentwicklung...")
    
    for idx, steps in enumerate(steps_range):
        print(f"Berechne Landschaft für {steps} Zeitschritte ({idx+1}/{len(steps_range)})")
        
        cost_map = np.zeros((grid_size, grid_size))
        
        for i, u in enumerate(u_values):
            for j, v in enumerate(v_values):
                cost_map[i, j] = simulate_pendulum(u, v, steps)
            
            if (i + 1) % 10 == 0:
                print(f"  Fortschritt: {i+1}/{grid_size}")
        
        # Ersetze inf-Werte für Visualisierung
        valid_costs = cost_map[np.isfinite(cost_map)]
        if len(valid_costs) > 0:
            max_finite = np.max(valid_costs)
            cost_map[~np.isfinite(cost_map)] = max_finite * 1.1
        else:
            cost_map[:] = 0
            
        landscapes.append(cost_map.T)
    
    return landscapes, U, V, steps_range

def create_animation(landscapes, U, V, steps_range):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Finde globale Min/Max für konsistente Farbgebung
    all_values = np.concatenate([landscape.flatten() for landscape in landscapes])
    vmin, vmax = np.min(all_values), np.max(all_values)
    
    def animate(frame):
        ax.clear()
        steps = steps_range[frame]
        Z = landscapes[frame]
        
        surf = ax.plot_surface(U, V, Z, cmap=cm.viridis, 
                              vmin=vmin, vmax=vmax, alpha=0.8)
        
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        ax.set_zlabel('Energy Dissipation')
        ax.set_title(f'Time Evolution of Stability Landscape\n'
                    f'Simulation Steps: {steps} (Time: {steps*dt:.2f}s)')
        ax.set_zlim(vmin, vmax)
        
        return surf,
    
    anim = FuncAnimation(fig, animate, frames=len(landscapes),
                        interval=300, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return anim

def analyze_landscape_evolution(landscapes, steps_range):
    """Analysiere die Entwicklung der Landschaft über die Zeit"""
    print("\nAnalyse der Landschaftsentwicklung:")
    
    # Berechne Metriken für jede Landschaft
    complexities = []
    min_energies = []
    
    for i, landscape in enumerate(landscapes):
        # Komplexität als Standardabweichung
        complexity = np.std(landscape)
        complexities.append(complexity)
        
        # Minimaler Energieverlust (ignoriere inf-Werte)
        valid_vals = landscape[np.isfinite(landscape)]
        if len(valid_vals) > 0:
            min_energy = np.min(valid_vals)
        else:
            min_energy = 0
        min_energies.append(min_energy)
        
        print(f"Steps {steps_range[i]}: Complexity={complexity:.4f}, Min Energy={min_energy:.4f}")
    
    # Plot der Entwicklung
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(steps_range, complexities, 'bo-')
    ax1.set_xlabel('Simulation Steps')
    ax1.set_ylabel('Landscape Complexity (Std. Dev.)')
    ax1.set_title('Evolution of Pattern Complexity')
    ax1.grid(True)
    
    ax2.plot(steps_range, min_energies, 'ro-')
    ax2.set_xlabel('Simulation Steps')
    ax2.set_ylabel('Minimum Energy Dissipation')
    ax2.set_title('Evolution of Minimum Energy Loss')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Hauptprogramm
if __name__ == "__main__":
    # Generiere alle Landschaften
    landscapes, U, V, steps_range = generate_animation_landscapes()
    
    # Erstelle Animation
    print("\nErstelle Animation...")
    anim = create_animation(landscapes, U, V, steps_range)
    
    # Analysiere die zeitliche Entwicklung
    analyze_landscape_evolution(landscapes, steps_range)
    
    # Speichere die Animation (optional)
    # print("Speichere Animation...")
    # anim.save('pendulum_landscape_evolution.gif', writer='pillow', fps=3)
    
    print("Fertig! Die Animation zeigt die Entwicklung der Stabilitätslandschaft über die Zeit.")

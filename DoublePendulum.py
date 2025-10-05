import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm

# Physikalische Parameter
g = 9.81  # Gravitationsbeschleunigung
damping = 0.08  # Erhöhte Dämpfung für Stabilität
L1, L2 = 1.0, 1.5  # Unterschiedliche Längen für stabilere Bewegung
m1, m2 = 1.3, 0.7  # Unterschiedliche Massen

# Simulationsparameter
u_range = 1.0  # Kleinere Range für stabilere Bewegung
v_range = 1.0
grid_size = 200  # Kleinere Gittergröße für schnellere Berechnung

# Berechne den Energieverlust für gegebene u,v-Werte
def simulate_pendulum(u, v, steps=50, dt=0.02):
    # Anfangsbedingungen basierend auf u,v (kleinere Variationen)
    theta1 = np.pi + u * 0.1
    theta2 = np.pi + v * 0.1
    theta1_dot = 0.0
    theta2_dot = 0.0
    
    # Energieberechnungsfunktion
    def calculate_energy(t1, t2, t1d, t2d):
        try:
            # Potentielle Energie
            y1 = -L1 * np.cos(t1)
            y2 = y1 - L2 * np.cos(t2)
            potential = m1 * g * y1 + m2 * g * y2
            
            # Kinetische Energie
            x1d = L1 * np.cos(t1) * t1d
            y1d = L1 * np.sin(t1) * t1d
            x2d = x1d + L2 * np.cos(t2) * t2d
            y2d = y1d + L2 * np.sin(t2) * t2d
            
            kinetic = 0.5 * m1 * (x1d**2 + y1d**2) + 0.5 * m2 * (x2d**2 + y2d**2)
            
            return potential + kinetic
        except:
            return 0.0
    
    # Simulation durchführen
    initial_energy = 0
    final_energy = 0
    
    for i in range(steps):
        try:
            # Berechne Winkelbeschleunigungen mit numerischer Stabilität
            delta = theta2 - theta1
            sin_delta = np.sin(delta)
            cos_delta = np.cos(delta)
            
            # Begrenze die Geschwindigkeiten für Stabilität
            theta1_dot = np.clip(theta1_dot, -10, 10)
            theta2_dot = np.clip(theta2_dot, -10, 10)
            
            denom1 = (m1 + m2) * L1 - m2 * L1 * cos_delta**2
            denom2 = (L2 / L1) * denom1
            
            # Vermeide Division durch Null
            if abs(denom1) < 1e-10 or abs(denom2) < 1e-10:
                break
                
            theta1_dot_dot = (m2 * L2 * theta2_dot**2 * sin_delta * cos_delta +
                             m2 * g * np.sin(theta2) * cos_delta +
                             m2 * L2 * theta2_dot**2 * sin_delta -
                             (m1 + m2) * g * np.sin(theta1)) / denom1
            
            theta2_dot_dot = (-m2 * L2 * theta2_dot**2 * sin_delta * cos_delta +
                             (m1 + m2) * (g * np.sin(theta1) * cos_delta - g * np.sin(theta2)) -
                             (m1 + m2) * L1 * theta1_dot**2 * sin_delta) / denom2
            
            # Dämpfung anwenden
            theta1_dot_dot -= damping * theta1_dot
            theta2_dot_dot -= damping * theta2_dot
            
            # Integration mit begrenzten Beschleunigungen
            theta1_dot_dot = np.clip(theta1_dot_dot, -50, 50)
            theta2_dot_dot = np.clip(theta2_dot_dot, -50, 50)
            
            theta1_dot += theta1_dot_dot * dt
            theta2_dot += theta2_dot_dot * dt
            theta1 += theta1_dot * dt
            theta2 += theta2_dot * dt
            
            # Energie am Anfang und Ende aufzeichnen
            if i == 5:
                initial_energy = calculate_energy(theta1, theta2, theta1_dot, theta2_dot)
            if i == steps - 5:
                final_energy = calculate_energy(theta1, theta2, theta1_dot, theta2_dot)
                
        except:
            # Bei Fehlern breche ab und gebe hohen Energieverlust zurück
            return float('inf')
    
    # Energieverlust ist unsere "Kosten"-Metrik
    energy_loss = abs(initial_energy - final_energy)
    return energy_loss if not np.isnan(energy_loss) else float('inf')

# Erzeuge die Kostenlandkarte im u,v-Raum
def generate_cost_map():
    cost_map = np.zeros((grid_size, grid_size))
    u_values = np.linspace(-u_range, u_range, grid_size)
    v_values = np.linspace(-v_range, v_range, grid_size)
    
    print("Berechne Energieverlust-Landschaft", end="", flush=True)
    
    for i, u in enumerate(u_values):
        for j, v in enumerate(v_values):
            cost_map[i, j] = simulate_pendulum(u, v)
            print(".", end="", flush=True)  # Fortschrittsanzeige
        
        # Zeige Fortschritt pro Zeile
        completed = (i + 1) * grid_size
        total = grid_size * grid_size
        print(f" {completed}/{total} Punkte")
    
    return cost_map, u_values, v_values

# Hauptprogramm
if __name__ == "__main__":
    # Kostenlandkarte berechnen
    cost_map, u_values, v_values = generate_cost_map()
    
    # Finde das Minimum in der Kostenlandkarte (ignoriere inf-Werte)
    valid_costs = cost_map[np.isfinite(cost_map)]
    if len(valid_costs) > 0:
        min_cost = np.min(valid_costs)
        min_idx = np.where(cost_map == min_cost)
        min_u = u_values[min_idx[0][0]]
        min_v = v_values[min_idx[1][0]]
        
        print(f"\nMinimum bei u={min_u:.3f}, v={min_v:.3f} mit Energieverlust={min_cost:.6f}")
    else:
        print("\nKeine gültigen Ergebnisse gefunden. Versuche andere Parameter.")
        min_u, min_v = 0, 0
    
    # Visualisierung der Energieverlust-Landschaft
    plt.figure(figsize=(10, 8))
    
    # Ersetze inf-Werte durch den maximalen endlichen Wert für die Visualisierung
    plot_data = np.copy(cost_map)
    plot_data[~np.isfinite(plot_data)] = np.max(plot_data[np.isfinite(plot_data)])
    
    plt.imshow(plot_data, extent=[-v_range, v_range, -u_range, u_range], 
               origin='lower', cmap=cm.Blues_r, aspect='auto')  # _r für umgekehrte Farbmap
    plt.colorbar(label='Energy Dissipation')
    
    if len(valid_costs) > 0:
        plt.scatter(min_v, min_u, color='red', marker='x', s=100, label='Minimum')
    
    plt.xlabel('v')
    plt.ylabel('u')
    plt.title('Energy dissipation landscape in u,v-Space\n(dark = low dissipation  = stable)')
    plt.legend()
    plt.show()
    
    # Zeige zusätzlich ein 3D-Plot für bessere Visualisierung
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    U, V = np.meshgrid(u_values, v_values)
    surf = ax.plot_surface(U, V, plot_data.T, cmap=cm.Blues_r, alpha=0.8)
    
    if len(valid_costs) > 0:
        ax.scatter(min_u, min_v, min_cost, color='red', s=100, label='Minimum')
    
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_zlabel('Energy Dissipation')
    ax.set_title('3D-Energy-dissipation-landscape')
    plt.legend()
    plt.show()
    
    # Einfache Pendelvisualisierung für das Minimum
    print(f"\nVisualisiere Pendelbewegung für u={min_u:.3f}, v={min_v:.3f}")
    
    # Simuliere die Bewegung
    theta1 = np.pi + min_u * 0.1
    theta2 = np.pi + min_v * 0.1
    theta1_dot = 0.0
    theta2_dot = 0.0
    dt = 0.02
    steps = 200
    
    positions = []
    
    for i in range(steps):
        # Physikberechnung (wie oben)
        delta = theta2 - theta1
        sin_delta = np.sin(delta)
        cos_delta = np.cos(delta)
        
        theta1_dot = np.clip(theta1_dot, -10, 10)
        theta2_dot = np.clip(theta2_dot, -10, 10)
        
        denom1 = (m1 + m2) * L1 - m2 * L1 * cos_delta**2
        denom2 = (L2 / L1) * denom1
        
        if abs(denom1) > 1e-10 and abs(denom2) > 1e-10:
            theta1_dot_dot = (m2 * L2 * theta2_dot**2 * sin_delta * cos_delta +
                             m2 * g * np.sin(theta2) * cos_delta +
                             m2 * L2 * theta2_dot**2 * sin_delta -
                             (m1 + m2) * g * np.sin(theta1)) / denom1
            
            theta2_dot_dot = (-m2 * L2 * theta2_dot**2 * sin_delta * cos_delta +
                             (m1 + m2) * (g * np.sin(theta1) * cos_delta - g * np.sin(theta2)) -
                             (m1 + m2) * L1 * theta1_dot**2 * sin_delta) / denom2
            
            theta1_dot_dot = np.clip(theta1_dot_dot, -50, 50)
            theta2_dot_dot = np.clip(theta2_dot_dot, -50, 50)
            
            theta1_dot += theta1_dot_dot * dt
            theta2_dot += theta2_dot_dot * dt
            theta1 += theta1_dot * dt
            theta2 += theta2_dot * dt
        
        # Speichere Positionen
        x1 = L1 * np.sin(theta1)
        y1 = -L1 * np.cos(theta1)
        x2 = x1 + L2 * np.sin(theta2)
        y2 = y1 - L2 * np.cos(theta2)
        
        positions.append((x1, y1, x2, y2))
    
    # Zeichne die Bewegung
    plt.figure(figsize=(10, 8))
    
    # Zeige die gesamte Trajektorie
    x1_vals = [p[0] for p in positions]
    y1_vals = [p[1] for p in positions]
    x2_vals = [p[2] for p in positions]
    y2_vals = [p[3] for p in positions]
    
    plt.plot(x1_vals, y1_vals, 'b-', alpha=0.3, label='Pendel 1 Spur')
    plt.plot(x2_vals, y2_vals, 'r-', alpha=0.3, label='Pendel 2 Spur')
    
    # Zeige die Endposition
    plt.plot([0, x1_vals[-1], x2_vals[-1]], [0, y1_vals[-1], y2_vals[-1]], 'o-', lw=2)
    
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.title(f'Pendelbewegung für u={min_u:.2f}, v={min_v:.2f}')
    plt.legend()
    plt.show()

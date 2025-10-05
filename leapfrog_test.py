import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time

class DoublePendulum:
    def __init__(self, m1=1.0, m2=1.0, L1=1.0, L2=1.0, c=0.001, g=9.81):
        self.m1 = m1
        self.m2 = m2
        self.L1 = L1
        self.L2 = L2
        self.c = c  # damping coefficient
        self.g = g
        
    def equations_of_motion(self, theta1, theta2, omega1, omega2):
        """
        Compute angular accelerations for the double pendulum system
        Returns [alpha1, alpha2]
        """
        # Common expressions
        cos_delta = np.cos(theta1 - theta2)
        sin_delta = np.sin(theta1 - theta2)
        sin_theta1 = np.sin(theta1)
        sin_theta2 = np.sin(theta2)
        
        # Matrix coefficients for the linear system
        a = (self.m1 + self.m2) * self.L1**2
        b = self.m2 * self.L1 * self.L2 * cos_delta
        c_val = self.m2 * self.L2**2
        d = b  # symmetric
        
        # Right-hand side terms
        f1 = (-self.m2 * self.L1 * self.L2 * omega2**2 * sin_delta 
              - (self.m1 + self.m2) * self.g * self.L1 * sin_theta1 
              - self.c * omega1)
        f2 = (self.m2 * self.L1 * self.L2 * omega1**2 * sin_delta 
              - self.m2 * self.g * self.L2 * sin_theta2 
              - self.c * omega2)
        
        # Solve linear system for angular accelerations
        determinant = a * c_val - b * d
        if abs(determinant) < 1e-12:
            # Handle near-singular matrix
            alpha1 = 0
            alpha2 = 0
        else:
            alpha1 = (c_val * f1 - b * f2) / determinant
            alpha2 = (a * f2 - d * f1) / determinant
        
        return alpha1, alpha2
    
    def total_energy(self, theta1, theta2, omega1, omega2):
        """Calculate total mechanical energy (kinetic + potential)"""
        # Kinetic energy
        v1_sq = (self.L1 * omega1)**2
        v2_sq = ((self.L1 * omega1)**2 + (self.L2 * omega2)**2 + 
                 2 * self.L1 * self.L2 * omega1 * omega2 * np.cos(theta1 - theta2))
        T = 0.5 * self.m1 * v1_sq + 0.5 * self.m2 * v2_sq
        
        # Potential energy
        y1 = -self.L1 * np.cos(theta1)
        y2 = y1 - self.L2 * np.cos(theta2)
        V = self.m1 * self.g * y1 + self.m2 * self.g * y2
        
        return T + V
    
    def leapfrog_integration(self, theta1, theta2, omega1, omega2, dt, steps):
        """
        Leapfrog (Verlet) integration for the double pendulum
        This method is symplectic and preserves energy well
        """
        # Initialize with a half step for velocities
        alpha1, alpha2 = self.equations_of_motion(theta1, theta2, omega1, omega2)
        omega1_half = omega1 + 0.5 * dt * alpha1
        omega2_half = omega2 + 0.5 * dt * alpha2
        
        # Store initial energy
        initial_energy = self.total_energy(theta1, theta2, omega1, omega2)
        
        # Main integration loop
        for i in range(steps):
            # Update positions (angles)
            theta1 += dt * omega1_half
            theta2 += dt * omega2_half
            
            # Update accelerations
            alpha1, alpha2 = self.equations_of_motion(theta1, theta2, omega1_half, omega2_half)
            
            # Update velocities
            omega1 = omega1_half + 0.5 * dt * alpha1
            omega2 = omega2_half + 0.5 * dt * alpha2
            
            # Prepare for next step
            omega1_half = omega1 + 0.5 * dt * alpha1
            omega2_half = omega2 + 0.5 * dt * alpha2
        
        final_energy = self.total_energy(theta1, theta2, omega1, omega2)
        return abs(final_energy - initial_energy)

def create_stability_landscape(m1=1.0, m2=1.0, L1=1.0, L2=1.0, c=0.001, steps=50, resolution=100):
    """
    Create stability landscape F(u,v) for parameter space using Leapfrog integration
    """
    pendulum = DoublePendulum(m1, m2, L1, L2, c)
    k = 0.2  # maximum deviation from stable position
    
    # Create parameter space grid
    u_values = np.linspace(-1, 1, resolution)
    v_values = np.linspace(-1, 1, resolution)
    U, V = np.meshgrid(u_values, v_values)
    F = np.zeros_like(U)
    
    print("Computing stability landscape using Leapfrog Integration...")
    print(f"Grid resolution: {resolution}x{resolution}")
    print(f"Time steps: {steps} (dt=0.01s, total time: {steps*0.01:.2f}s)")
    start_time = time.time()
    
    completed = 0
    total_points = len(u_values) * len(v_values)
    
    for i in range(len(u_values)):
        for j in range(len(v_values)):
            # Map (u,v) to initial conditions
            theta1_0 = np.pi + U[j,i] * k
            theta2_0 = np.pi + V[j,i] * k
            omega1_0 = 0.0
            omega2_0 = 0.0
            
            # Simulate and store energy dissipation using Leapfrog
            F[j,i] = pendulum.leapfrog_integration(
                theta1_0, theta2_0, omega1_0, omega2_0, 
                dt=0.01, steps=steps
            )
            completed += 1
        
        if (i + 1) % 20 == 0:
            progress = (completed / total_points) * 100
            elapsed = time.time() - start_time
            eta = (elapsed / progress) * (100 - progress) if progress > 0 else 0
            print(f"Progress: {i + 1}/{len(u_values)} ({progress:.1f}%) - "
                  f"Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s")
    
    total_time = time.time() - start_time
    print(f"Total computation time: {total_time:.2f} seconds")
    print(f"Average time per point: {total_time/total_points*1000:.2f} ms")
    
    return U, V, F

def plot_stability_landscape(U, V, F, m1, m2, L1, L2, c, steps):
    """Plot the stability landscape"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 2D contour plot
    im = ax1.contourf(U, V, F, levels=50, cmap='viridis')
    ax1.set_xlabel('Parameter u', fontsize=12)
    ax1.set_ylabel('Parameter v', fontsize=12)
    ax1.set_title(f'2D Energy Dissipation Landscape\nm1={m1}, m2={m2}, L1={L1}, L2={L2}, c={c}', fontsize=13)
    ax1.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax1, label='Energy Dissipation F(u,v)')
    
    # 3D surface plot
    from mpl_toolkits.mplot3d import Axes3D
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(U, V, F, cmap='viridis', alpha=0.8, 
                           linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax2.set_xlabel('Parameter u', fontsize=12)
    ax2.set_ylabel('Parameter v', fontsize=12)
    ax2.set_zlabel('Energy Dissipation F(u,v)', fontsize=12)
    ax2.set_title(f'3D Stability Landscape\n{steps} time steps (Leapfrog integration)', fontsize=13)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=20, label='Energy Dissipation')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def analyze_patterns(F):
    """Analyze the geometric patterns in the stability landscape"""
    print("\n" + "="*50)
    print("PATTERN ANALYSIS")
    print("="*50)
    
    # Basic statistics
    print(f"Energy Dissipation Statistics:")
    print(f"  Mean: {np.mean(F):.6f} J")
    print(f"  Std:  {np.std(F):.6f} J")
    print(f"  Min:  {np.min(F):.6f} J")
    print(f"  Max:  {np.max(F):.6f} J")
    print(f"  Range:{np.max(F) - np.min(F):.6f} J")
    
    # Pattern analysis
    threshold_low = np.percentile(F, 25)  # Low dissipation valleys
    threshold_high = np.percentile(F, 75)  # High dissipation ridges
    
    valley_mask = F < threshold_low
    ridge_mask = F > threshold_high
    
    valley_fraction = np.sum(valley_mask) / F.size
    ridge_fraction = np.sum(ridge_mask) / F.size
    
    # Calculate structure complexity
    from scipy import ndimage
    labeled, num_features = ndimage.label(valley_mask)
    print(f"\nPattern Features:")
    print(f"  Valley regions (low dissipation): {valley_fraction:.1%}")
    print(f"  Ridge regions (high dissipation): {ridge_fraction:.1%}")
    print(f"  Number of distinct valley structures: {num_features}")
    print(f"  Structure complexity: {'High' if num_features > 20 else 'Medium' if num_features > 10 else 'Low'}")
    
    return valley_mask, ridge_mask

def compare_integration_methods(m1=1.0, m2=1.0, L1=1.0, L2=1.0, c=0.001, steps=50, test_points=5):
    """Compare Leapfrog with Euler method for validation"""
    print("\n" + "="*50)
    print("INTEGRATION METHOD COMPARISON")
    print("="*50)
    
    pendulum = DoublePendulum(m1, m2, L1, L2, c)
    k = 0.2
    
    # Test at random points in parameter space
    np.random.seed(42)
    test_u = np.random.uniform(-1, 1, test_points)
    test_v = np.random.uniform(-1, 1, test_points)
    
    print("Comparing Leapfrog vs Euler methods at test points:")
    print("Point\tu\tv\tLeapfrog\tEuler\t\tDifference")
    print("-" * 60)
    
    leapfrog_results = []
    euler_results = []
    
    for i in range(test_points):
        theta1_0 = np.pi + test_u[i] * k
        theta2_0 = np.pi + test_v[i] * k
        
        # Leapfrog
        leapfrog_energy = pendulum.leapfrog_integration(
            theta1_0, theta2_0, 0, 0, 0.01, steps
        )
        
        # Euler (for comparison)
        euler_energy = pendulum.euler_integration(
            theta1_0, theta2_0, 0, 0, 0.01, steps
        )
        
        leapfrog_results.append(leapfrog_energy)
        euler_results.append(euler_energy)
        
        diff = abs(leapfrog_energy - euler_energy)
        print(f"{i+1}\t{test_u[i]:.2f}\t{test_v[i]:.2f}\t{leapfrog_energy:.6f}\t{euler_energy:.6f}\t{diff:.6f}")
    
    avg_leapfrog = np.mean(leapfrog_results)
    avg_euler = np.mean(euler_results)
    print(f"\nAverage energy dissipation:")
    print(f"  Leapfrog: {avg_leapfrog:.6f}")
    print(f"  Euler:    {avg_euler:.6f}")
    print(f"  Ratio:    {avg_leapfrog/avg_euler:.3f}")

# Adding Euler method for comparison
def euler_integration(self, theta1_0, theta2_0, omega1_0, omega2_0, dt, steps):
    """Euler integration for comparison"""
    theta1, theta2, omega1, omega2 = theta1_0, theta2_0, omega1_0, omega2_0
    initial_energy = self.total_energy(theta1, theta2, omega1, omega2)
    
    for _ in range(steps):
        alpha1, alpha2 = self.equations_of_motion(theta1, theta2, omega1, omega2)
        omega1 += dt * alpha1
        omega2 += dt * alpha2
        theta1 += dt * omega1
        theta2 += dt * omega2
    
    final_energy = self.total_energy(theta1, theta2, omega1, omega2)
    return abs(final_energy - initial_energy)

# Add Euler method to the class
DoublePendulum.euler_integration = euler_integration

def run_simulation_set():
    """Run multiple simulations with different parameters"""
    simulation_sets = [
        # Simulation 1a from study
        {'m1': 1.0, 'm2': 1.0, 'L1': 1.0, 'L2': 1.0, 'c': 0.001, 'steps': 50, 'name': 'Simulation 1a (Symmetric)'},
        
        # Simulation with different masses
        {'m1': 1.3, 'm2': 0.7, 'L1': 1.0, 'L2': 1.5, 'c': 0.001, 'steps': 50, 'name': 'Simulation 2a (Asymmetric)'},
        
        # Simulation with higher damping
        {'m1': 1.0, 'm2': 1.0, 'L1': 1.0, 'L2': 1.0, 'c': 0.004, 'steps': 50, 'name': 'High Damping'},
    ]
    
    for params in simulation_sets:
        print(f"\n{'='*60}")
        print(f"RUNNING: {params['name']}")
        print(f"{'='*60}")
        
        U, V, F = create_stability_landscape(
            m1=params['m1'], m2=params['m2'], 
            L1=params['L1'], L2=params['L2'], 
            c=params['c'], steps=params['steps'],
            resolution=80  # Lower resolution for quicker testing
        )
        
        fig = plot_stability_landscape(U, V, F, 
                               params['m1'], params['m2'], 
                               params['L1'], params['L2'], 
                               params['c'], params['steps'])
        
        valley_mask, ridge_mask = analyze_patterns(F)
        
        # Save figure
        filename = f"leapfrog_{params['name'].replace(' ', '_').replace('(', '').replace(')', '')}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {filename}")

def main():
    """Main function to run the simulation"""
    print("DOUBLE PENDULUM TRANSIENT STABILITY ANALYSIS")
    print("Using Leapfrog (Verlet) Integration - Symplectic Method")
    print("=" * 60)
    
    # Choose between single simulation or set
    choice = input("Run (1) Single simulation, (2) Simulation set, or (3) Method comparison? [1/2/3]: ").strip()
    
    if choice == "2":
        run_simulation_set()
        return
    elif choice == "3":
        compare_integration_methods()
        return
    
    # Single simulation parameters
    print("\nEnter simulation parameters (press Enter for defaults):")
    
    m1 = float(input("Mass 1 (kg) [1.0]: ") or "1.0")
    m2 = float(input("Mass 2 (kg) [1.0]: ") or "1.0")
    L1 = float(input("Length 1 (m) [1.0]: ") or "1.0")
    L2 = float(input("Length 2 (m) [1.0]: ") or "1.0")
    c = float(input("Damping coefficient [0.001]: ") or "0.001")
    steps = int(input("Time steps [50]: ") or "50")
    resolution = int(input("Grid resolution [80]: ") or "80")
    
    print(f"\nSimulation Parameters:")
    print(f"  m1={m1}, m2={m2}, L1={L1}, L2={L2}")
    print(f"  damping c={c}, steps={steps}, resolution={resolution}x{resolution}")
    
    # Create stability landscape
    U, V, F = create_stability_landscape(m1, m2, L1, L2, c, steps, resolution)
    
    # Plot results
    fig = plot_stability_landscape(U, V, F, m1, m2, L1, L2, c, steps)
    
    # Analyze patterns
    valley_mask, ridge_mask = analyze_patterns(F)
    
    # Save figure option
    save = input("\nSave figure? [y/N]: ").strip().lower()
    if save == 'y':
        filename = f"leapfrog_stability_m1_{m1}_m2_{m2}_steps_{steps}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {filename}")

if __name__ == "__main__":
    main()

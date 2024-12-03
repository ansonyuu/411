import numpy as np
from scipy.optimize import minimize

class SimpleDroneOptimizer:
    def __init__(self,
                 M_frame,           # Mass of frame (kg)
                 C_frame,           # Cost of frame
                 battery_cost_per_kg,  # Cost per kg of battery
                 battery_energy_density # Wh/kg
                 ):
        self.M_frame = M_frame
        self.C_frame = C_frame
        self.battery_cost_per_kg = battery_cost_per_kg
        self.battery_energy_density = battery_energy_density

    def objective(self, x):
        """
        Objective function to minimize
        x[0] = Battery mass (kg)
        """
        M_battery = x[0]
        
        # Calculate total cost
        total_cost = self.C_frame + (M_battery * self.battery_cost_per_kg)
        
        # Calculate flight time (simplified model)
        total_mass = self.M_frame + M_battery
        flight_time = (self.battery_energy_density * M_battery) / total_mass
        
        # Return negative of weighted sum (since we're minimizing)
        return total_cost - 50 * flight_time  # Adjust weight (50) to balance cost vs flight time

    def optimize(self, initial_battery_mass=1.0):
        # Simple bounds: battery mass must be positive and less than twice frame mass
        bounds = [(0, self.M_frame * 2)]
        
        # Run optimization
        result = minimize(
            self.objective,
            [initial_battery_mass],
            method='SLSQP',
            bounds=bounds
        )
        
        if result.success:
            M_battery = result.x[0]
            total_mass = self.M_frame + M_battery
            flight_time = (self.battery_energy_density * M_battery) / total_mass
            total_cost = self.C_frame + (M_battery * self.battery_cost_per_kg)
            
            return {
                'status': 'optimal',
                'battery_mass': M_battery,
                'total_mass': total_mass,
                'flight_time': flight_time,
                'total_cost': total_cost
            }
        else:
            return {'status': 'not feasible', 'message': result.message}

# Example usage
if __name__ == "__main__":
    # Example constants
    constants = {
        'M_frame': 2.0,              # Frame mass in kg
        'C_frame': 1000,             # Frame cost
        'battery_cost_per_kg': 500,  # Cost per kg of battery
        'battery_energy_density': 200 # Wh/kg for lithium battery
    }
    
    # Create optimizer and run optimization
    optimizer = SimpleDroneOptimizer(**constants)
    result = optimizer.optimize()
    
    # Print results
    if result['status'] == 'optimal':
        print("\nOptimal Solution Found:")
        print(f"Battery Mass: {result['battery_mass']:.2f} kg")
        print(f"Total Mass: {result['total_mass']:.2f} kg")
        print(f"Flight Time: {result['flight_time']:.2f} minutes")
        print(f"Total Cost: {result['total_cost']:.2f}")
    else:
        print("No feasible solution found")
        print(f"Message: {result['message']}")
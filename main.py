import numpy as np
from scipy.optimize import minimize

class DroneOptimizer:
    def __init__(self,
                 C_frame,
                 C_electronics,
                 M_frame,
                 M_electronics,
                 k_battery,
                 Th,
                 b,
                 battery_energy_density,
                 prop_efficiency,
                 prop_count,
                 min_battery_capacity,
                 max_battery_capacity,
                 min_flight_time,
                 max_total_weight
                 ):
        self.C_frame = C_frame
        self.C_electronics = C_electronics
        self.M_frame = M_frame
        self.M_electronics = M_electronics
        self.k_battery = k_battery
        self.Th = Th
        self.b = b
        self.battery_energy_density = battery_energy_density
        self.prop_efficiency = prop_efficiency
        self.prop_count = prop_count
        self.min_battery_capacity = min_battery_capacity
        self.max_battery_capacity = max_battery_capacity
        self.min_flight_time = min_flight_time
        self.max_total_weight = max_total_weight

    def objective(self, x):
        C_battery = x[0]
        total_cost = self.C_frame + C_battery + self.C_electronics
        flight_time = self.calculate_flight_time(x)
        return total_cost - 50 * flight_time  # Adjust weight (50) to balance cost vs flight time

    def calculate_mass(self, x):
        C_battery = x[0]
        M_battery = self.k_battery * C_battery * self.Th + self.k_battery * C_battery + self.b
        return self.M_frame + self.M_electronics + M_battery

    def calculate_flight_time(self, x):
        C_battery = x[0]
        M_battery = self.k_battery * C_battery * self.Th + self.k_battery * C_battery + self.b
        total_mass = self.calculate_mass(x)
        power_factor = self.prop_count * (1/self.prop_efficiency)
        flight_time = (self.battery_energy_density * M_battery) / (power_factor * total_mass)
        return flight_time

    def constraints(self, x):
        C_battery = x[0]
        total_mass = self.calculate_mass(x)
        flight_time = self.calculate_flight_time(x)
        return [
            self.max_total_weight - total_mass,
            flight_time - self.min_flight_time,
            C_battery - self.min_battery_capacity,
            self.max_battery_capacity - C_battery
        ]

    def optimize(self):
        x0 = [20]
        bounds = [(self.min_battery_capacity, self.max_battery_capacity)]
        cons = [{'type': 'ineq', 'fun': lambda x, i=i: self.constraints(x)[i]} for i in range(4)]
        
        result = minimize(
            self.objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'ftol': 1e-6, 'maxiter': 1000}
        )
        
        if result.success:
            C_battery = result.x[0]
            total_mass = self.calculate_mass(result.x)
            flight_time = self.calculate_flight_time(result.x)
            total_cost = self.C_frame + C_battery + self.C_electronics
            M_battery = self.k_battery * C_battery * self.Th + self.k_battery * C_battery + self.b
            
            return {
                'status': 'optimal',
                'battery_capacity': C_battery,
                'battery_mass': M_battery,
                'total_mass': total_mass,
                'flight_time': flight_time,
                'total_cost': total_cost
            }
        else:
            return {
                'status': 'not feasible',
                'message': result.message
            }

# Example usage
if __name__ == "__main__":
    constants = {
        'C_frame': 1000,
        'C_electronics': 500,
        'M_frame': 2.0,
        'M_electronics': 1.0,
        'k_battery': 0.1,
        'Th': 1.01,
        'b': 0.1,
        'battery_energy_density': 250,
        'prop_efficiency': 0.8,
        'prop_count': 4,
        'min_battery_capacity': 16,
        'max_battery_capacity': 32,
        'min_flight_time': 10,
        'max_total_weight': 10.0
    }
    
    optimizer = DroneOptimizer(**constants)
    result = optimizer.optimize()
    
    if result['status'] == 'optimal':
        print("\nOptimal Solution Found:")
        print(f"Battery Capacity: {result['battery_capacity']:.2f} Ah")
        print(f"Battery Mass: {result['battery_mass']:.2f} kg")
        print(f"Flight Time: {result['flight_time']:.2f} minutes")
        print(f"Total Cost: {result['total_cost']:.2f}")
        print(f"Total Mass: {result['total_mass']:.2f} kg")
    else:
        print("No feasible solution found")
        print(f"Message: {result['message']}")
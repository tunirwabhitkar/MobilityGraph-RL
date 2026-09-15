from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np

class VehicleRouter:
    """Solves the Vehicle Routing Problem (VRP) to physically move the vehicles."""
    
    def __init__(self, num_vehicles: int, depot_index: int = 0):
        self.num_vehicles = num_vehicles
        self.depot = depot_index
        
    def solve(self, dist_matrix: np.ndarray, pickups: list, deliveries: list):
        """
        pickups and deliveries are mapped from the relocations array.
        Each relocation from i to j becomes a pickup at i and delivery at j.
        """
        # Convert relocations into a format suitable for VRP
        # A simple approximation: just visit the nodes that require interaction
        
        # We need a proper distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            # multiply by 1000 to convert to int for OR-Tools, assuming dist_matrix has small floats
            return int(dist_matrix[from_node][to_node] * 1000)

        num_nodes = len(dist_matrix)
        manager = pywrapcp.RoutingIndexManager(num_nodes, self.num_vehicles, self.depot)
        routing = pywrapcp.RoutingModel(manager)
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add Distance constraint to limit maximum route length
        dimension_name = 'Distance'
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            30000,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name)
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)
        
        # Pickup and Delivery constraints
        for pickup, delivery in zip(pickups, deliveries):
            pickup_index = manager.NodeToIndex(pickup)
            delivery_index = manager.NodeToIndex(delivery)
            routing.AddPickupAndDelivery(pickup_index, delivery_index)
            routing.solver().Add(
                routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index))
            routing.solver().Add(
                distance_dimension.CumulVar(pickup_index) <= distance_dimension.CumulVar(delivery_index))
                
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
            
        solution = routing.SolveWithParameters(search_parameters)
        
        routes = []
        if solution:
            for vehicle_id in range(self.num_vehicles):
                index = routing.Start(vehicle_id)
                route = []
                while not routing.IsEnd(index):
                    route.append(manager.IndexToNode(index))
                    index = solution.Value(routing.NextVar(index))
                route.append(manager.IndexToNode(index))
                if len(route) > 2: # More than just Start -> End
                    routes.append(route)
                    
        return routes

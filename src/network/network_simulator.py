"""
Network Simulator for 5G Edge Computing Environment
"""

import asyncio
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ..utils.logger import setup_logger


@dataclass
class EdgeNode:
    """Represents an edge computing node"""
    node_id: str
    location: Tuple[float, float]  # (latitude, longitude)
    capacity: int  # Maximum number of clients
    current_load: int  # Current number of connected clients
    processing_power: float  # GFLOPS
    status: str  # "active", "maintenance", "overloaded"
    last_heartbeat: datetime


@dataclass
class NetworkCondition:
    """Network condition parameters"""
    bandwidth_mbps: float
    latency_ms: float
    packet_loss: float
    jitter_ms: float
    signal_strength: float  # dBm


class NetworkSimulator:
    """
    Simulates 5G network conditions and edge node infrastructure
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = setup_logger("NetworkSimulator")
        
        # Network parameters
        self.base_bandwidth = self.config.get("bandwidth_mbps", 1000)
        self.base_latency = self.config.get("latency_ms", 1)
        self.base_packet_loss = self.config.get("packet_loss", 0.001)
        
        # Edge nodes registry
        self.edge_nodes: Dict[str, EdgeNode] = {}
        self.edge_server_instances: Dict[str, Any] = {}  # server_id -> EdgeServer instance
        self.device_connections: Dict[str, str] = {}  # device_id -> node_id
        
        # Network conditions cache
        self.network_conditions_cache: Dict[str, NetworkCondition] = {}
        
        # Only initialize default nodes if explicitly requested
        if self.config.get("initialize_default_nodes", False):
            self._initialize_edge_nodes()
        
        self.logger.info("Network simulator initialized")
    
    def _initialize_edge_nodes(self):
        """Initialize edge computing nodes"""
        # Simulate edge nodes in a metropolitan area
        node_locations = [
            (40.7128, -74.0060),  # NYC Manhattan
            (40.7614, -73.9776),  # NYC Times Square
            (40.6892, -74.0445),  # NYC Brooklyn
            (40.7282, -73.7949),  # NYC Queens
            (40.8176, -73.9782),  # NYC Bronx
        ]
        
        for i, location in enumerate(node_locations):
            node_id = f"edge_node_{i+1}"
            self.edge_nodes[node_id] = EdgeNode(
                node_id=node_id,
                location=location,
                capacity=50,
                current_load=0,  # Start with 0 load instead of random
                processing_power=random.uniform(100, 500),  # GFLOPS
                status="active",
                last_heartbeat=datetime.now()
            )
        
        self.logger.info(f"Initialized {len(self.edge_nodes)} edge nodes")
    
    def clear_default_nodes(self):
        """Clear default edge nodes to use only registered EdgeServers"""
        default_nodes = [f"edge_node_{i+1}" for i in range(5)]
        for node_id in default_nodes:
            if node_id in self.edge_nodes and node_id not in self.edge_server_instances:
                del self.edge_nodes[node_id]
        self.logger.info(f"Cleared default nodes, remaining: {list(self.edge_nodes.keys())}")
    
    async def discover_edge_nodes(
        self, 
        device_location: Tuple[float, float],
        signal_threshold: float = -80
    ) -> List[Dict[str, Any]]:
        """
        Discover available edge nodes within range
        
        Args:
            device_location: Device location (lat, lon)
            signal_threshold: Minimum signal strength in dBm
            
        Returns:
            List of available edge nodes with their properties
        """
        available_nodes = []
        
        self.logger.debug(f"Discovering edge nodes from {device_location}")
        self.logger.debug(f"Available edge nodes: {list(self.edge_nodes.keys())}")
        self.logger.debug(f"EdgeServer instances: {list(self.edge_server_instances.keys())}")
        
        for node_id, node in self.edge_nodes.items():
            # Calculate distance
            distance = self._calculate_distance(device_location, node.location)
            
            # Calculate signal strength based on distance
            signal_strength = self._calculate_signal_strength(distance)
            
            # Check if node is reachable and available
            if (signal_strength >= signal_threshold and 
                node.status == "active" and 
                node.current_load < node.capacity):
                
                # Prioritize actual EdgeServer instances
                is_real_server = node_id in self.edge_server_instances
                priority_boost = 100 if is_real_server else 0
                
                available_nodes.append({
                    "node_id": node_id,
                    "location": node.location,
                    "distance": distance,
                    "signal_strength": signal_strength + priority_boost,  # Boost real servers
                    "load": (node.current_load / node.capacity) * 100,
                    "processing_power": node.processing_power,
                    "available_capacity": node.capacity - node.current_load,
                    "is_real_server": is_real_server
                })
                
                self.logger.debug(f"Found node {node_id}: distance={distance:.1f}m, "
                                f"signal={signal_strength:.1f}dBm, real_server={is_real_server}")
        
        # Sort by signal strength and real server status
        available_nodes.sort(
            key=lambda x: (x["signal_strength"], -x["load"]), 
            reverse=True
        )
        
        self.logger.info(f"Discovery complete: found {len(available_nodes)} available nodes")
        
        return available_nodes
    
    def _calculate_distance(
        self, 
        loc1: Tuple[float, float], 
        loc2: Tuple[float, float]
    ) -> float:
        """Calculate distance between two locations in meters"""
        lat1, lon1 = loc1
        lat2, lon2 = loc2
        
        # Haversine formula
        R = 6371000  # Earth's radius in meters
        
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * 
             np.sin(dlon/2)**2)
        
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        
        return distance
    
    def _calculate_signal_strength(self, distance: float) -> float:
        """
        Calculate 5G signal strength based on distance
        
        Args:
            distance: Distance in meters
            
        Returns:
            Signal strength in dBm
        """
        # Simplified 5G path loss model optimized for simulation
        # More lenient than real-world conditions for demonstration
        
        if distance < 10:
            distance = 10  # Minimum distance
        
        # Free space path loss at 3.5 GHz (5G mid-band)
        frequency_ghz = 3.5
        fspl = 20 * np.log10(distance) + 20 * np.log10(frequency_ghz) + 92.45
        
        # Reduced shadowing and fading for more predictable simulation
        shadowing = random.gauss(0, 3)  # Reduced from 8 to 3
        fading = random.gauss(0, 1)     # Reduced from 3 to 1
        
        # Higher base station transmit power for better coverage
        tx_power_dbm = 50  # Increased from 46 to 50 dBm
        
        signal_strength = tx_power_dbm - fspl + shadowing + fading
        
        # Ensure minimum signal strength for devices within reasonable range
        if distance < 5000:  # Within 5km, guarantee good signal
            signal_strength = max(signal_strength, -70)
        elif distance < 10000:  # Within 10km, guarantee usable signal
            signal_strength = max(signal_strength, -85)
        
        return signal_strength
    
    async def connect_to_edge_node(
        self, 
        device_id: str, 
        node_id: str,
        device_public_key: bytes = None
    ) -> Dict[str, Any]:
        """
        Connect a device to an edge node
        
        Args:
            device_id: Device identifier
            node_id: Target edge node identifier
            device_public_key: Device public key for secure communication
            
        Returns:
            Connection result
        """
        if node_id not in self.edge_nodes:
            return {"success": False, "error": "Edge node not found"}
        
        node = self.edge_nodes[node_id]
        
        # Check capacity
        if node.current_load >= node.capacity:
            return {"success": False, "error": "Edge node at capacity"}
        
        # Check node status
        if node.status != "active":
            return {"success": False, "error": f"Edge node status: {node.status}"}
        
        # Simulate connection establishment delay
        connection_delay = random.uniform(0.1, 0.5)  # 100-500ms
        await asyncio.sleep(connection_delay)
        
        # Update connection state
        if device_id in self.device_connections:
            # Disconnect from previous node
            old_node_id = self.device_connections[device_id]
            if old_node_id in self.edge_nodes:
                self.edge_nodes[old_node_id].current_load -= 1
                # Disconnect from actual EdgeServer instance
                if old_node_id in self.edge_server_instances:
                    old_server = self.edge_server_instances[old_node_id]
                    await old_server.disconnect_client(device_id, reason="reconnection")
        
        self.device_connections[device_id] = node_id
        node.current_load += 1
        
        # Connect to actual EdgeServer instance
        connection_success = True
        if node_id in self.edge_server_instances:
            edge_server = self.edge_server_instances[node_id]
            client_info = {
                "device_id": device_id,
                "public_key": device_public_key,
                "connection_time": datetime.now(),
                "capabilities": {
                    "training": True,
                    "inference": True,
                    "high_performance": False
                }
            }
            connection_success = await edge_server.connect_client(device_id, client_info)
        
        if not connection_success:
            # Rollback if EdgeServer connection failed
            self.device_connections.pop(device_id, None)
            node.current_load -= 1
            return {"success": False, "error": "EdgeServer connection failed"}
        
        self.logger.info(f"Device {device_id} connected to {node_id}")
        
        return {
            "success": True,
            "node_id": node_id,
            "connection_time": connection_delay,
            "assigned_resources": {
                "bandwidth_mbps": self.base_bandwidth / max(1, node.current_load),
                "compute_allocation": node.processing_power / max(1, node.current_load)
            }
        }
    
    async def send_model_update(
        self, 
        device_id: str, 
        node_id: str, 
        encrypted_data: bytes
    ) -> bool:
        """
        Send model update from device to edge node
        
        Args:
            device_id: Source device ID
            node_id: Target edge node ID
            encrypted_data: Encrypted model data
            
        Returns:
            Success status
        """
        if device_id not in self.device_connections:
            self.logger.warning(f"Device {device_id} not connected to any edge node")
            return False
        
        if self.device_connections[device_id] != node_id:
            self.logger.warning(f"Device {device_id} not connected to {node_id}")
            return False
        
        # Get network conditions
        network_condition = await self._get_network_conditions(device_id, node_id)
        
        # Calculate transmission parameters
        data_size_mb = len(encrypted_data) / (1024 * 1024)
        transmission_time = (data_size_mb * 8) / network_condition.bandwidth_mbps  # seconds
        
        # Simulate packet loss
        if random.random() < network_condition.packet_loss:
            self.logger.warning(f"Packet loss during transmission from {device_id}")
            return False
        
        # Simulate transmission delay
        total_delay = network_condition.latency_ms / 1000 + transmission_time
        await asyncio.sleep(total_delay)
        
        # Actually deliver the model update to the edge server
        if node_id in self.edge_server_instances:
            server_instance = self.edge_server_instances[node_id]
            success = await server_instance.receive_model_update(device_id, encrypted_data)
            if not success:
                self.logger.warning(f"Edge server {node_id} rejected update from {device_id}")
                return False
        else:
            self.logger.warning(f"No server instance found for {node_id}")
            return False
        
        self.logger.info(
            f"Model update delivered from {device_id} to {node_id} "
            f"({data_size_mb:.2f}MB in {total_delay:.3f}s)"
        )
        
        return True
    
    async def broadcast_global_model(
        self, 
        node_id: str, 
        encrypted_model_data: bytes,
        target_devices: List[str]
    ) -> Dict[str, bool]:
        """
        Broadcast global model from edge node to connected devices
        
        Args:
            node_id: Source edge node ID
            encrypted_model_data: Encrypted global model
            target_devices: List of target device IDs
            
        Returns:
            Delivery status for each device
        """
        results = {}
        
        for device_id in target_devices:
            if (device_id in self.device_connections and 
                self.device_connections[device_id] == node_id):
                
                # Get network conditions
                network_condition = await self._get_network_conditions(device_id, node_id)
                
                # Calculate transmission parameters
                data_size_mb = len(encrypted_model_data) / (1024 * 1024)
                transmission_time = (data_size_mb * 8) / network_condition.bandwidth_mbps
                
                # Simulate transmission
                if random.random() >= network_condition.packet_loss:
                    total_delay = network_condition.latency_ms / 1000 + transmission_time
                    await asyncio.sleep(total_delay)
                    results[device_id] = True
                    
                    self.logger.info(
                        f"Global model sent to {device_id} "
                        f"({data_size_mb:.2f}MB in {total_delay:.3f}s)"
                    )
                else:
                    results[device_id] = False
                    self.logger.warning(f"Failed to send global model to {device_id}")
            else:
                results[device_id] = False
                self.logger.warning(f"Device {device_id} not connected to {node_id}")
        
        return results
    
    async def _get_network_conditions(
        self, 
        device_id: str, 
        node_id: str
    ) -> NetworkCondition:
        """
        Get current network conditions between device and edge node
        """
        cache_key = f"{device_id}_{node_id}"
        
        # Check cache (simulate dynamic network conditions)
        if (cache_key in self.network_conditions_cache and 
            random.random() < 0.8):  # 80% cache hit rate
            return self.network_conditions_cache[cache_key]
        
        # Calculate new network conditions
        if node_id in self.edge_nodes:
            node = self.edge_nodes[node_id]
            
            # Base conditions affected by node load
            load_factor = node.current_load / node.capacity
            
            # Bandwidth decreases with load
            bandwidth = self.base_bandwidth * (1 - 0.5 * load_factor)
            
            # Latency increases with load
            latency = self.base_latency * (1 + 2 * load_factor)
            
            # Add random variations
            bandwidth += random.gauss(0, bandwidth * 0.1)
            latency += random.gauss(0, latency * 0.2)
            
            # Ensure reasonable bounds
            bandwidth = max(10, min(bandwidth, self.base_bandwidth))
            latency = max(0.5, latency)
            
            conditions = NetworkCondition(
                bandwidth_mbps=bandwidth,
                latency_ms=latency,
                packet_loss=self.base_packet_loss * (1 + load_factor),
                jitter_ms=random.uniform(0.1, 2.0),
                signal_strength=random.uniform(-60, -40)  # Good 5G signal
            )
        else:
            # Default poor conditions if node not found
            conditions = NetworkCondition(
                bandwidth_mbps=10,
                latency_ms=100,
                packet_loss=0.1,
                jitter_ms=10,
                signal_strength=-90
            )
        
        # Cache the conditions
        self.network_conditions_cache[cache_key] = conditions
        
        return conditions
    
    def disconnect_device(self, device_id: str):
        """Disconnect a device from its edge node"""
        if device_id in self.device_connections:
            node_id = self.device_connections[device_id]
            if node_id in self.edge_nodes:
                self.edge_nodes[node_id].current_load -= 1
                # Disconnect from actual EdgeServer instance
                if node_id in self.edge_server_instances:
                    server = self.edge_server_instances[node_id]
                    # Note: This should be called with await in async context
                    # For now, we'll just update the server's connected_clients directly
                    if hasattr(server, 'connected_clients') and device_id in server.connected_clients:
                        server.connected_clients.remove(device_id)
                        if device_id in server.client_states:
                            del server.client_states[device_id]
            
            del self.device_connections[device_id]
            self.logger.info(f"Device {device_id} disconnected from {node_id}")
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get network performance statistics"""
        total_devices = len(self.device_connections)
        
        node_utilization = {}
        for node_id, node in self.edge_nodes.items():
            node_utilization[node_id] = {
                "load_percentage": (node.current_load / node.capacity) * 100,
                "connected_devices": node.current_load,
                "capacity": node.capacity,
                "status": node.status
            }
        
        return {
            "total_connected_devices": total_devices,
            "total_edge_nodes": len(self.edge_nodes),
            "node_utilization": node_utilization,
            "network_conditions": {
                "base_bandwidth_mbps": self.base_bandwidth,
                "base_latency_ms": self.base_latency,
                "base_packet_loss": self.base_packet_loss
            }
        }
    
    def register_edge_server(self, server_id: str, location: Tuple[float, float], capacity: int = 50, server_instance = None):
        """Register an EdgeServer with the NetworkSimulator for discovery"""
        self.edge_nodes[server_id] = EdgeNode(
            node_id=server_id,
            location=location,
            capacity=capacity,
            current_load=0,  # Will be updated when devices connect
            processing_power=10.0,  # GFLOPS
            status="active",
            last_heartbeat=datetime.now()
        )
        
        # Store reference to actual EdgeServer instance
        if server_instance:
            self.edge_server_instances[server_id] = server_instance
        
        self.logger.info(f"Registered edge server {server_id} at location {location}")
    
    def unregister_edge_server(self, server_id: str):
        """Unregister an EdgeServer from the NetworkSimulator"""
        if server_id in self.edge_nodes:
            del self.edge_nodes[server_id]
        if server_id in self.edge_server_instances:
            del self.edge_server_instances[server_id]
        self.logger.info(f"Unregistered edge server {server_id}")
    
    def clear_default_nodes(self):
        """Clear any default edge nodes that aren't actual EdgeServer instances"""
        nodes_to_remove = []
        for node_id in self.edge_nodes.keys():
            if node_id not in self.edge_server_instances:
                nodes_to_remove.append(node_id)
        
        for node_id in nodes_to_remove:
            del self.edge_nodes[node_id]
            self.logger.info(f"Removed default edge node {node_id}")

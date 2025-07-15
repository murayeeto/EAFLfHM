"""
Edge Client Manager
Manages client connections, capabilities, and resource allocation
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import uuid

from ..utils.logger import setup_logger


@dataclass
class ClientInfo:
    """Client information and capabilities"""
    client_id: str
    device_type: str
    hardware_specs: Dict[str, Any]
    battery_level: float
    signal_strength: float
    location: tuple
    connected_at: datetime
    last_seen: datetime
    capabilities: Dict[str, Any]


@dataclass
class ResourceAllocation:
    """Resource allocation for a client"""
    client_id: str
    allocated_bandwidth: float  # Mbps
    allocated_compute: float    # GFLOPS
    priority_level: int         # 1-5, higher is better
    quality_of_service: str     # "high", "medium", "low"


class EdgeClientManager:
    """
    Manages client connections and resource allocation at edge servers
    """
    
    def __init__(self, max_clients: int = 50):
        self.logger = setup_logger("EdgeClientManager")
        
        # Configuration
        self.max_clients = max_clients
        
        # Client management
        self.clients: Dict[str, ClientInfo] = {}
        self.resource_allocations: Dict[str, ResourceAllocation] = {}
        self.client_groups: Dict[str, Set[str]] = defaultdict(set)  # Group clients by criteria
        
        # Performance tracking
        self.client_performance = defaultdict(lambda: {
            "response_times": [],
            "data_quality_scores": [],
            "reliability_score": 1.0,
            "participation_count": 0
        })
        
        # Resource limits
        self.total_bandwidth = 1000.0  # Mbps
        self.total_compute = 500.0     # GFLOPS
        self.allocated_bandwidth = 0.0
        self.allocated_compute = 0.0
        
        self.logger.info(f"Client manager initialized with capacity for {max_clients} clients")
    
    async def register_client(self, client_id: str, client_info: Dict[str, Any]) -> bool:
        """
        Register a new client with the edge server
        
        Args:
            client_id: Unique client identifier
            client_info: Client information dictionary
            
        Returns:
            True if registration successful
        """
        try:
            # Check capacity
            if len(self.clients) >= self.max_clients:
                self.logger.warning(f"Cannot register client {client_id}: at capacity")
                return False
            
            # Check if already registered
            if client_id in self.clients:
                self.logger.warning(f"Client {client_id} already registered")
                return True
            
            # Create client info object
            client_info_obj = ClientInfo(
                client_id=client_id,
                device_type=client_info.get("device_type", "unknown"),
                hardware_specs=client_info.get("hardware_specs", {}),
                battery_level=client_info.get("battery_level", 100.0),
                signal_strength=client_info.get("signal_strength", -50.0),
                location=client_info.get("location", (0.0, 0.0)),
                connected_at=datetime.now(),
                last_seen=datetime.now(),
                capabilities=client_info.get("capabilities", {})
            )
            
            # Register client
            self.clients[client_id] = client_info_obj
            
            # Allocate resources
            await self._allocate_resources(client_id)
            
            # Group client
            self._group_client(client_id, client_info_obj)
            
            self.logger.info(
                f"Client {client_id} registered - "
                f"Type: {client_info_obj.device_type}, "
                f"Battery: {client_info_obj.battery_level:.1f}%"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering client {client_id}: {e}")
            return False
    
    async def unregister_client(self, client_id: str):
        """
        Unregister a client and free resources
        
        Args:
            client_id: Client identifier to unregister
        """
        try:
            if client_id not in self.clients:
                self.logger.warning(f"Client {client_id} not registered")
                return
            
            # Free allocated resources
            await self._deallocate_resources(client_id)
            
            # Remove from groups
            self._ungroup_client(client_id)
            
            # Remove client record
            client_info = self.clients[client_id]
            connection_duration = (datetime.now() - client_info.connected_at).total_seconds()
            
            del self.clients[client_id]
            
            # Keep performance history for a while
            # (can be used for re-connecting clients)
            
            self.logger.info(
                f"Client {client_id} unregistered - "
                f"Connection duration: {connection_duration:.0f}s"
            )
            
        except Exception as e:
            self.logger.error(f"Error unregistering client {client_id}: {e}")
    
    async def update_client_status(
        self, 
        client_id: str, 
        status_update: Dict[str, Any]
    ) -> bool:
        """
        Update client status information
        
        Args:
            client_id: Client identifier
            status_update: Status update dictionary
            
        Returns:
            True if update successful
        """
        try:
            if client_id not in self.clients:
                self.logger.warning(f"Cannot update status for unregistered client {client_id}")
                return False
            
            client_info = self.clients[client_id]
            
            # Update fields
            if "battery_level" in status_update:
                client_info.battery_level = status_update["battery_level"]
            
            if "signal_strength" in status_update:
                client_info.signal_strength = status_update["signal_strength"]
            
            if "location" in status_update:
                client_info.location = status_update["location"]
            
            # Always update last seen
            client_info.last_seen = datetime.now()
            
            # Check if resource reallocation is needed
            await self._check_resource_reallocation(client_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating client status {client_id}: {e}")
            return False
    
    async def _allocate_resources(self, client_id: str):
        """Allocate resources to a client"""
        try:
            client_info = self.clients[client_id]
            
            # Calculate resource allocation based on client characteristics
            priority = self._calculate_client_priority(client_info)
            
            # Base allocation
            base_bandwidth = min(20.0, self.total_bandwidth / max(1, len(self.clients)))
            base_compute = min(10.0, self.total_compute / max(1, len(self.clients)))
            
            # Adjust based on priority and availability
            priority_multiplier = 0.5 + (priority / 5.0)  # 0.7-1.2 range
            
            allocated_bandwidth = min(
                base_bandwidth * priority_multiplier,
                self.total_bandwidth - self.allocated_bandwidth
            )
            
            allocated_compute = min(
                base_compute * priority_multiplier,
                self.total_compute - self.allocated_compute
            )
            
            # Ensure minimum allocation
            allocated_bandwidth = max(1.0, allocated_bandwidth)
            allocated_compute = max(0.5, allocated_compute)
            
            # Determine QoS level
            qos = self._determine_qos_level(client_info, priority)
            
            # Create resource allocation
            allocation = ResourceAllocation(
                client_id=client_id,
                allocated_bandwidth=allocated_bandwidth,
                allocated_compute=allocated_compute,
                priority_level=priority,
                quality_of_service=qos
            )
            
            self.resource_allocations[client_id] = allocation
            
            # Update totals
            self.allocated_bandwidth += allocated_bandwidth
            self.allocated_compute += allocated_compute
            
            self.logger.info(
                f"Resources allocated to {client_id} - "
                f"BW: {allocated_bandwidth:.1f}Mbps, "
                f"Compute: {allocated_compute:.1f}GFLOPS, "
                f"QoS: {qos}"
            )
            
        except Exception as e:
            self.logger.error(f"Error allocating resources to {client_id}: {e}")
    
    async def _deallocate_resources(self, client_id: str):
        """Deallocate resources from a client"""
        try:
            if client_id in self.resource_allocations:
                allocation = self.resource_allocations[client_id]
                
                # Free resources
                self.allocated_bandwidth -= allocation.allocated_bandwidth
                self.allocated_compute -= allocation.allocated_compute
                
                # Ensure non-negative
                self.allocated_bandwidth = max(0, self.allocated_bandwidth)
                self.allocated_compute = max(0, self.allocated_compute)
                
                del self.resource_allocations[client_id]
                
                self.logger.info(f"Resources deallocated from {client_id}")
            
        except Exception as e:
            self.logger.error(f"Error deallocating resources from {client_id}: {e}")
    
    def _calculate_client_priority(self, client_info: ClientInfo) -> int:
        """Calculate client priority (1-5, higher is better)"""
        priority = 3  # Base priority
        
        # Battery level factor
        if client_info.battery_level > 80:
            priority += 1
        elif client_info.battery_level < 20:
            priority -= 1
        
        # Signal strength factor
        if client_info.signal_strength > -60:  # Strong signal
            priority += 1
        elif client_info.signal_strength < -80:  # Weak signal
            priority -= 1
        
        # Device capability factor
        if client_info.capabilities.get("high_performance", False):
            priority += 1
        
        # Historical performance factor
        if client_info.client_id in self.client_performance:
            perf = self.client_performance[client_info.client_id]
            if perf["reliability_score"] > 0.8:
                priority += 1
            elif perf["reliability_score"] < 0.5:
                priority -= 1
        
        return max(1, min(5, priority))
    
    def _determine_qos_level(self, client_info: ClientInfo, priority: int) -> str:
        """Determine Quality of Service level for client"""
        if priority >= 4:
            return "high"
        elif priority >= 3:
            return "medium"
        else:
            return "low"
    
    def _group_client(self, client_id: str, client_info: ClientInfo):
        """Group client based on characteristics"""
        # Group by device type
        self.client_groups[f"device_{client_info.device_type}"].add(client_id)
        
        # Group by battery level
        if client_info.battery_level > 80:
            self.client_groups["high_battery"].add(client_id)
        elif client_info.battery_level > 50:
            self.client_groups["medium_battery"].add(client_id)
        else:
            self.client_groups["low_battery"].add(client_id)
        
        # Group by signal strength
        if client_info.signal_strength > -60:
            self.client_groups["strong_signal"].add(client_id)
        elif client_info.signal_strength > -80:
            self.client_groups["medium_signal"].add(client_id)
        else:
            self.client_groups["weak_signal"].add(client_id)
        
        # Group by QoS level
        if client_id in self.resource_allocations:
            qos = self.resource_allocations[client_id].quality_of_service
            self.client_groups[f"qos_{qos}"].add(client_id)
    
    def _ungroup_client(self, client_id: str):
        """Remove client from all groups"""
        for group_clients in self.client_groups.values():
            group_clients.discard(client_id)
    
    async def _check_resource_reallocation(self, client_id: str):
        """Check if resource reallocation is needed for a client"""
        try:
            if client_id not in self.clients or client_id not in self.resource_allocations:
                return
            
            client_info = self.clients[client_id]
            current_allocation = self.resource_allocations[client_id]
            
            # Recalculate priority
            new_priority = self._calculate_client_priority(client_info)
            
            # If priority changed significantly, reallocate
            if abs(new_priority - current_allocation.priority_level) >= 2:
                await self._deallocate_resources(client_id)
                await self._allocate_resources(client_id)
                
                # Update grouping
                self._ungroup_client(client_id)
                self._group_client(client_id, client_info)
        
        except Exception as e:
            self.logger.error(f"Error checking resource reallocation for {client_id}: {e}")
    
    def get_clients_by_group(self, group_name: str) -> List[str]:
        """Get list of client IDs in a specific group"""
        return list(self.client_groups.get(group_name, set()))
    
    def get_client_info(self, client_id: str) -> Optional[ClientInfo]:
        """Get client information"""
        return self.clients.get(client_id)
    
    def get_resource_allocation(self, client_id: str) -> Optional[ResourceAllocation]:
        """Get resource allocation for a client"""
        return self.resource_allocations.get(client_id)
    
    def select_clients_for_round(
        self, 
        selection_criteria: Dict[str, Any] = None
    ) -> List[str]:
        """
        Select clients for a federated learning round
        
        Args:
            selection_criteria: Criteria for client selection
            
        Returns:
            List of selected client IDs
        """
        criteria = selection_criteria or {}
        
        # Default selection parameters
        max_clients = criteria.get("max_clients", min(10, len(self.clients)))
        min_battery = criteria.get("min_battery", 20.0)
        min_signal = criteria.get("min_signal", -90.0)
        prefer_high_qos = criteria.get("prefer_high_qos", True)
        
        # Filter eligible clients
        eligible_clients = []
        
        for client_id, client_info in self.clients.items():
            # Check basic criteria
            if (client_info.battery_level >= min_battery and
                client_info.signal_strength >= min_signal):
                
                # Check if client was recently active
                time_since_last_seen = (datetime.now() - client_info.last_seen).total_seconds()
                if time_since_last_seen < 300:  # Active within last 5 minutes
                    eligible_clients.append(client_id)
        
        # Sort by priority if preferring high QoS
        if prefer_high_qos and eligible_clients:
            def priority_key(client_id):
                allocation = self.resource_allocations.get(client_id)
                if allocation:
                    return allocation.priority_level
                return 0
            
            eligible_clients.sort(key=priority_key, reverse=True)
        
        # Select up to max_clients
        selected_clients = eligible_clients[:max_clients]
        
        self.logger.info(
            f"Selected {len(selected_clients)} clients from {len(eligible_clients)} eligible "
            f"(total: {len(self.clients)})"
        )
        
        return selected_clients
    
    def update_client_performance(
        self, 
        client_id: str, 
        response_time: float,
        data_quality_score: float = 1.0
    ):
        """Update client performance metrics"""
        if client_id not in self.clients:
            return
        
        perf = self.client_performance[client_id]
        
        # Update response times (keep last 20)
        perf["response_times"].append(response_time)
        if len(perf["response_times"]) > 20:
            perf["response_times"] = perf["response_times"][-15:]
        
        # Update data quality scores (keep last 20)
        perf["data_quality_scores"].append(data_quality_score)
        if len(perf["data_quality_scores"]) > 20:
            perf["data_quality_scores"] = perf["data_quality_scores"][-15:]
        
        # Update participation count
        perf["participation_count"] += 1
        
        # Recalculate reliability score
        avg_response_time = sum(perf["response_times"]) / len(perf["response_times"])
        avg_quality = sum(perf["data_quality_scores"]) / len(perf["data_quality_scores"])
        
        # Simple reliability calculation (can be made more sophisticated)
        response_factor = max(0.1, min(1.0, 1.0 - (avg_response_time - 1.0) / 10.0))
        quality_factor = avg_quality
        participation_factor = min(1.0, perf["participation_count"] / 10.0)
        
        perf["reliability_score"] = (response_factor + quality_factor + participation_factor) / 3.0
    
    def get_client_statistics(self) -> Dict[str, Any]:
        """Get comprehensive client statistics"""
        if not self.clients:
            return {"error": "No clients registered"}
        
        # Basic statistics
        total_clients = len(self.clients)
        
        # Battery statistics
        battery_levels = [client.battery_level for client in self.clients.values()]
        avg_battery = sum(battery_levels) / len(battery_levels)
        
        # Signal strength statistics
        signal_strengths = [client.signal_strength for client in self.clients.values()]
        avg_signal = sum(signal_strengths) / len(signal_strengths)
        
        # Resource utilization
        bandwidth_utilization = (self.allocated_bandwidth / self.total_bandwidth) * 100
        compute_utilization = (self.allocated_compute / self.total_compute) * 100
        
        # Group statistics
        group_stats = {
            group_name: len(client_ids) 
            for group_name, client_ids in self.client_groups.items()
            if client_ids
        }
        
        return {
            "total_clients": total_clients,
            "max_capacity": self.max_clients,
            "utilization_percent": (total_clients / self.max_clients) * 100,
            "average_battery_level": avg_battery,
            "average_signal_strength": avg_signal,
            "bandwidth_utilization_percent": bandwidth_utilization,
            "compute_utilization_percent": compute_utilization,
            "group_distribution": group_stats,
            "total_allocated_bandwidth": self.allocated_bandwidth,
            "total_allocated_compute": self.allocated_compute
        }

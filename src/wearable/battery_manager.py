"""
Battery Manager for Wearable Devices
Simulates battery consumption and power management
"""

import time
import math
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..utils.logger import setup_logger


@dataclass
class PowerConsumptionEvent:
    """Record of power consumption event"""
    timestamp: datetime
    component: str
    power_watts: float
    duration_seconds: float
    energy_consumed_wh: float


class BatteryManager:
    """
    Manages battery simulation for wearable devices
    """
    
    def __init__(self, capacity_mah: float = 300, voltage: float = 3.7):
        """
        Initialize battery manager
        
        Args:
            capacity_mah: Battery capacity in mAh
            voltage: Battery voltage in volts
        """
        self.logger = setup_logger("BatteryManager")
        
        # Battery specifications
        self.capacity_mah = capacity_mah
        self.voltage = voltage
        self.capacity_wh = (capacity_mah * voltage) / 1000  # Convert to Wh
        
        # Battery state
        self.current_charge_wh = self.capacity_wh  # Start fully charged
        self.charge_cycles = 0
        self.health_factor = 1.0  # Battery degradation factor (1.0 = new)
        
        # Power consumption tracking
        self.consumption_history = []
        self.last_update_time = time.time()
        
        # Power management modes
        self.power_mode = "normal"  # normal, power_save, ultra_save
        self.thermal_throttling = False
        self.temperature_celsius = 25.0
        
        # Component power consumption baselines (watts)
        self.base_power_consumption = {
            "cpu_idle": 0.050,      # 50mW
            "cpu_active": 0.200,    # 200mW
            "cpu_training": 0.500,  # 500mW during ML training
            "display": 0.100,       # 100mW
            "wireless": 0.080,      # 80mW for 5G/WiFi
            "sensors": 0.016,       # 16mW total for all sensors
            "storage": 0.010,       # 10mW
            "other": 0.020          # 20mW for other components
        }
        
        self.logger.info(
            f"Battery manager initialized - Capacity: {capacity_mah}mAh "
            f"({self.capacity_wh:.2f}Wh)"
        )
    
    def consume_power(self, power_watts: float, duration_seconds: float = 1.0, component: str = "unknown"):
        """
        Consume power from battery
        
        Args:
            power_watts: Power consumption in watts
            duration_seconds: Duration of consumption
            component: Component consuming power
        """
        current_time = time.time()
        
        # Apply power management modifications
        adjusted_power = self._apply_power_management(power_watts)
        
        # Calculate energy consumed
        energy_consumed_wh = (adjusted_power * duration_seconds) / 3600  # Convert to Wh
        
        # Apply battery health factor
        effective_consumption = energy_consumed_wh / self.health_factor
        
        # Update battery charge
        self.current_charge_wh -= effective_consumption
        self.current_charge_wh = max(0, self.current_charge_wh)  # Can't go below 0
        
        # Record consumption event
        event = PowerConsumptionEvent(
            timestamp=datetime.now(),
            component=component,
            power_watts=adjusted_power,
            duration_seconds=duration_seconds,
            energy_consumed_wh=effective_consumption
        )
        self.consumption_history.append(event)
        
        # Limit history size
        if len(self.consumption_history) > 1000:
            self.consumption_history = self.consumption_history[-800:]
        
        # Update temperature based on power consumption
        self._update_temperature(adjusted_power)
        
        # Check for low battery conditions
        self._check_battery_conditions()
        
        self.last_update_time = current_time
    
    def _apply_power_management(self, base_power: float) -> float:
        """Apply power management optimizations"""
        power = base_power
        
        # Power mode adjustments
        if self.power_mode == "power_save":
            power *= 0.7  # 30% reduction
        elif self.power_mode == "ultra_save":
            power *= 0.5  # 50% reduction
        
        # Thermal throttling
        if self.thermal_throttling:
            power *= 0.8  # 20% reduction when overheating
        
        return power
    
    def _update_temperature(self, power_watts: float):
        """Update device temperature based on power consumption"""
        # Simple thermal model
        ambient_temp = 25.0  # Celsius
        thermal_resistance = 20.0  # K/W (device to ambient)
        
        # Temperature rise due to power consumption
        temp_rise = power_watts * thermal_resistance
        target_temp = ambient_temp + temp_rise
        
        # Thermal time constant (exponential approach to target)
        tau = 300  # seconds (5 minutes)
        dt = time.time() - self.last_update_time
        alpha = 1 - math.exp(-dt / tau)
        
        self.temperature_celsius = (
            self.temperature_celsius * (1 - alpha) + 
            target_temp * alpha
        )
        
        # Check for thermal throttling
        if self.temperature_celsius > 40.0 and not self.thermal_throttling:
            self.thermal_throttling = True
            self.logger.warning(f"Thermal throttling activated - Temp: {self.temperature_celsius:.1f}°C")
        elif self.temperature_celsius < 35.0 and self.thermal_throttling:
            self.thermal_throttling = False
            self.logger.info("Thermal throttling deactivated")
    
    def _check_battery_conditions(self):
        """Check for low battery and other conditions"""
        battery_percentage = self.get_battery_level()
        
        if battery_percentage <= 5 and battery_percentage > 0:
            if self.power_mode != "ultra_save":
                self.set_power_mode("ultra_save")
                self.logger.critical(f"Critical battery level: {battery_percentage:.1f}%")
        elif battery_percentage <= 20:
            if self.power_mode == "normal":
                self.set_power_mode("power_save")
                self.logger.warning(f"Low battery level: {battery_percentage:.1f}%")
        elif battery_percentage > 30:
            if self.power_mode != "normal":
                self.set_power_mode("normal")
                self.logger.info("Battery level sufficient - returning to normal mode")
    
    def get_battery_level(self) -> float:
        """
        Get current battery level as percentage
        
        Returns:
            Battery level (0-100%)
        """
        effective_capacity = self.capacity_wh * self.health_factor
        return max(0, min(100, (self.current_charge_wh / effective_capacity) * 100))
    
    def get_remaining_time_hours(self, current_power_w: float = None) -> float:
        """
        Estimate remaining battery time
        
        Args:
            current_power_w: Current power consumption in watts
            
        Returns:
            Estimated remaining time in hours
        """
        if current_power_w is None:
            # Estimate based on recent consumption
            recent_events = [
                event for event in self.consumption_history[-20:]
                if (datetime.now() - event.timestamp).seconds < 300  # Last 5 minutes
            ]
            
            if recent_events:
                avg_power = sum(event.power_watts for event in recent_events) / len(recent_events)
            else:
                avg_power = self.base_power_consumption["cpu_idle"]  # Conservative estimate
        else:
            avg_power = current_power_w
        
        if avg_power <= 0:
            return float('inf')
        
        return self.current_charge_wh / avg_power
    
    def set_power_mode(self, mode: str):
        """
        Set power management mode
        
        Args:
            mode: "normal", "power_save", or "ultra_save"
        """
        if mode in ["normal", "power_save", "ultra_save"]:
            old_mode = self.power_mode
            self.power_mode = mode
            self.logger.info(f"Power mode changed from {old_mode} to {mode}")
        else:
            self.logger.error(f"Invalid power mode: {mode}")
    
    def charge_battery(self, charge_rate_w: float = 5.0, duration_seconds: float = 1.0):
        """
        Simulate battery charging
        
        Args:
            charge_rate_w: Charging power in watts
            duration_seconds: Charging duration
        """
        # Convert charging power to energy
        energy_added_wh = (charge_rate_w * duration_seconds) / 3600
        
        # Apply charging efficiency (typically 85-90%)
        efficiency = 0.87
        effective_energy = energy_added_wh * efficiency
        
        # Update battery charge
        max_charge = self.capacity_wh * self.health_factor
        self.current_charge_wh = min(max_charge, self.current_charge_wh + effective_energy)
        
        # Charging generates heat
        self._update_temperature(charge_rate_w * 0.3)  # 30% of charge power becomes heat
    
    def simulate_battery_aging(self, cycles: int = 1):
        """
        Simulate battery aging and capacity degradation
        
        Args:
            cycles: Number of charge cycles to simulate
        """
        self.charge_cycles += cycles
        
        # Typical lithium-ion degradation: 20% capacity loss after 500 cycles
        degradation_rate = 0.0004  # 0.04% per cycle
        self.health_factor = max(0.6, 1.0 - (self.charge_cycles * degradation_rate))
        
        if cycles > 0:
            self.logger.info(
                f"Battery aging: {self.charge_cycles} cycles, "
                f"health: {self.health_factor:.1%}"
            )
    
    def get_power_consumption_summary(self, hours: int = 1) -> Dict[str, Any]:
        """
        Get power consumption summary for the last N hours
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Power consumption summary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_events = [
            event for event in self.consumption_history 
            if event.timestamp > cutoff_time
        ]
        
        if not recent_events:
            return {"error": "No consumption data available"}
        
        # Aggregate by component
        component_consumption = {}
        total_energy_wh = 0
        
        for event in recent_events:
            if event.component not in component_consumption:
                component_consumption[event.component] = 0
            component_consumption[event.component] += event.energy_consumed_wh
            total_energy_wh += event.energy_consumed_wh
        
        # Calculate average power
        total_duration_h = sum(event.duration_seconds for event in recent_events) / 3600
        avg_power_w = total_energy_wh / max(0.001, total_duration_h)
        
        return {
            "period_hours": hours,
            "total_energy_wh": total_energy_wh,
            "average_power_w": avg_power_w,
            "component_breakdown": component_consumption,
            "battery_level_change": len(recent_events) * 0.1,  # Rough estimate
            "events_count": len(recent_events)
        }
    
    def get_battery_status(self) -> Dict[str, Any]:
        """Get comprehensive battery status"""
        return {
            "battery_level_percent": self.get_battery_level(),
            "current_charge_wh": self.current_charge_wh,
            "capacity_wh": self.capacity_wh,
            "health_factor": self.health_factor,
            "charge_cycles": self.charge_cycles,
            "power_mode": self.power_mode,
            "temperature_celsius": self.temperature_celsius,
            "thermal_throttling": self.thermal_throttling,
            "estimated_remaining_hours": self.get_remaining_time_hours(),
            "voltage": self.voltage,
            "last_update": datetime.fromtimestamp(self.last_update_time).isoformat()
        }
    
    def reset_battery(self, charge_level: float = 100.0):
        """
        Reset battery to specified charge level
        
        Args:
            charge_level: Target charge level (0-100%)
        """
        charge_level = max(0, min(100, charge_level))
        self.current_charge_wh = (charge_level / 100) * self.capacity_wh * self.health_factor
        self.consumption_history.clear()
        self.logger.info(f"Battery reset to {charge_level:.1f}%")

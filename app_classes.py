from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict

@dataclass
class UserSettings:
    theme: str = "dark"  # dark, light, auto
    favorites: List[str] = field(default_factory=list)
    export_presets: Dict[str, Dict] = field(default_factory=dict)
    performance_metrics: bool = True
    keyboard_shortcuts: bool = True

@dataclass
class PerformanceMetrics:
    api_call_times: List[float] = field(default_factory=list)
    cache_hit_rate: float = 0.0
    last_sync_duration: float = 0.0
    avg_response_time: float = 0.0

class Theme(Enum):
    DARK = "dark"
    LIGHT = "light"
    AUTO = "auto"

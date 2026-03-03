"""Agents module exports."""

from .base import ReflexionAgent
from .optimized import OptimizedReflexionAgent
from .original import OriginalReflexionAgent
from .smart import SmartReflexionAgent
from .vector import VectorReflexionAgent  # ✅ ALREADY EXPORTED
from .multiagent import MultiAgentReflexion  # ✅ FIXED: Only main class

__all__ = [
    'ReflexionAgent', 
    'OptimizedReflexionAgent', 
    'OriginalReflexionAgent', 
    'SmartReflexionAgent',
    'VectorReflexionAgent',  # ✅ ALREADY EXPORTED
    'MultiAgentReflexion'    # ✅ FIXED: Remove SharedReflectionPool
]

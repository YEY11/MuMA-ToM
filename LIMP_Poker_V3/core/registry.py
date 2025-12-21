"""
LIMP_Poker_V3 Agent Registry
Supports pluggable agent architecture for ablation experiments
"""

from typing import Dict, Type, List, Any, Optional
from loguru import logger


class AgentRegistry:
    """
    Central registry for all agents.
    Supports dynamic registration and configuration-based instantiation.
    """

    _perception_agents: Dict[str, Type] = {}
    _reasoning_agents: Dict[str, Type] = {}

    # ========== Registration Decorators ==========

    @classmethod
    def register_perception(cls, name: str):
        """
        Decorator to register a perception layer agent.

        Usage:
            @AgentRegistry.register_perception("board_agent")
            class BoardAgent:
                ...
        """

        def decorator(agent_class: Type) -> Type:
            cls._perception_agents[name] = agent_class
            logger.debug(f"Registered perception agent: {name}")
            return agent_class

        return decorator

    @classmethod
    def register_reasoning(cls, name: str):
        """
        Decorator to register a reasoning layer agent.

        Usage:
            @AgentRegistry.register_reasoning("tom_belief_agent")
            class TomBeliefAgent:
                ...
        """

        def decorator(agent_class: Type) -> Type:
            cls._reasoning_agents[name] = agent_class
            logger.debug(f"Registered reasoning agent: {name}")
            return agent_class

        return decorator

    # ========== Agent Retrieval ==========

    @classmethod
    def get_perception_agents(cls, agent_config: Dict[str, bool]) -> List[Any]:
        """
        Instantiate and return all enabled perception agents.

        Args:
            agent_config: Dict mapping agent names to enabled status

        Returns:
            List of instantiated agent objects
        """
        agents = []
        for name, enabled in agent_config.items():
            if enabled and name in cls._perception_agents:
                try:
                    agent = cls._perception_agents[name]()
                    agents.append(agent)
                    logger.debug(f"Instantiated perception agent: {name}")
                except Exception as e:
                    logger.error(f"Failed to instantiate {name}: {e}")
        return agents

    @classmethod
    def get_reasoning_agents(cls, agent_config: Dict[str, bool]) -> List[Any]:
        """
        Instantiate and return all enabled reasoning agents.

        Args:
            agent_config: Dict mapping agent names to enabled status

        Returns:
            List of instantiated agent objects
        """
        agents = []
        for name, enabled in agent_config.items():
            if enabled and name in cls._reasoning_agents:
                try:
                    agent = cls._reasoning_agents[name]()
                    agents.append(agent)
                    logger.debug(f"Instantiated reasoning agent: {name}")
                except Exception as e:
                    logger.error(f"Failed to instantiate {name}: {e}")
        return agents

    @classmethod
    def get_agent_by_name(
        cls, name: str, layer: str = "auto"
    ) -> Optional[Type]:
        """
        Get agent class by name.

        Args:
            name: Agent name
            layer: "perception", "reasoning", or "auto" (search both)

        Returns:
            Agent class or None if not found
        """
        if layer in ("perception", "auto"):
            if name in cls._perception_agents:
                return cls._perception_agents[name]

        if layer in ("reasoning", "auto"):
            if name in cls._reasoning_agents:
                return cls._reasoning_agents[name]

        return None

    # ========== Introspection ==========

    @classmethod
    def list_registered(cls) -> Dict[str, List[str]]:
        """List all registered agents by layer"""
        return {
            "perception": list(cls._perception_agents.keys()),
            "reasoning": list(cls._reasoning_agents.keys()),
        }

    @classmethod
    def print_registry(cls):
        """Print registered agents for debugging"""
        print("=" * 40)
        print("Agent Registry")
        print("=" * 40)
        print("Perception Agents:")
        for name in cls._perception_agents:
            print(f"  - {name}")
        print("Reasoning Agents:")
        for name in cls._reasoning_agents:
            print(f"  - {name}")
        print("=" * 40)


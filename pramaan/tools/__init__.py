"""The five investigation tools, in the order the agent usually calls them."""
from agent_core import ToolRegistry

from . import check_domain, detect_red_flags, extract_entities, verify_vacancy, web_reputation

ALL_TOOLS = [
    extract_entities.TOOL,
    detect_red_flags.TOOL,
    check_domain.TOOL,
    verify_vacancy.TOOL,
    web_reputation.TOOL,
]


def build_registry() -> ToolRegistry:
    return ToolRegistry(ALL_TOOLS)

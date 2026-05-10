"""
research/tasks/__init__.py
Automated research tasks for quant strategy discovery.
"""

from .auto_quant_research import AutoResearchConfig, run_auto_quant_research

__all__ = ["AutoResearchConfig", "run_auto_quant_research"]

"""Graph presentation types shared by tensor networks and Feynman diagrams."""

from ..graph_native import DiagramRender, LayoutSettings, StrokeStyle, initialize_module

initialize_module()
del initialize_module

__all__ = ["DiagramRender", "LayoutSettings", "StrokeStyle"]

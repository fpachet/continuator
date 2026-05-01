"""Optional UI integrations for Continuator."""

__all__ = ["Continuator_gradio"]


def __getattr__(name):
    if name == "Continuator_gradio":
        from ctor.ui.gradio_app import Continuator_gradio

        return Continuator_gradio
    raise AttributeError(name)

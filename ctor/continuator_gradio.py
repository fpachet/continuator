"""
Compatibility entry point for the local Gradio UI.

The UI implementation lives in ctor.ui.gradio_app. This module is kept so
existing commands such as `python -m ctor.continuator_gradio` keep working.
"""

from ctor.ui.gradio_app import Continuator_gradio


if __name__ == "__main__":
    Continuator_gradio().launch()

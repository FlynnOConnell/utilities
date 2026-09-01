"""Where the viewer is running, and what that changes.

A notebook kernel has no window of its own: the canvas goes in the output
cell through jupyter_rfb, the kernel already runs an event loop, and any
native dialog would open on the machine the kernel runs on, which for a
JupyterHub session is not the machine the user is looking at. Everything
that branches on that asks :func:`in_notebook`, not ``get_ipython()``: a
plain ``ipython`` terminal has an IPython shell but no kernel, and wants a
desktop window like any other terminal.
"""

from __future__ import annotations

__all__ = ["in_notebook", "display_widget"]


def in_notebook() -> bool:
    """True inside a Jupyter kernel (lab, notebook, VS Code, hub); False in a
    terminal, a plain ``ipython`` shell, a script or pytest."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    if shell is None:
        return False
    # ZMQInteractiveShell is the kernel; TerminalInteractiveShell is `ipython`.
    # The kernel attribute is the duck-typed version for shells that subclass.
    return shell.__class__.__name__ == "ZMQInteractiveShell" or hasattr(
        shell, "kernel"
    )


def display_widget(output) -> None:
    """Put a shown canvas in the current output cell.

    ``Figure.show()`` returns the ipywidget in a notebook; it renders only if
    it is displayed, which happens for a cell's last expression or through
    ``IPython.display.display``. ``run_gui`` is a statement, not an
    expression, so it displays explicitly.
    """
    if output is None:
        return
    try:
        from IPython.display import display
    except ImportError:
        return
    display(output)

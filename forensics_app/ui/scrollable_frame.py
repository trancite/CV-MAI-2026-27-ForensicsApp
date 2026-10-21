"""A vertically scrollable container for the tools sidebar."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class ScrollableFrame(ttk.Frame):
    """Expose ``content`` as the parent for a column of sidebar widgets."""

    def __init__(self, master: tk.Misc, **kwargs: object) -> None:
        kwargs.setdefault("style", "Sidebar.TFrame")
        super().__init__(master, **kwargs)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            self,
            background="#eef1f5",
            borderwidth=0,
            highlightthickness=0,
            yscrollincrement=20,
        )
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.content = ttk.Frame(self.canvas, style="Sidebar.TFrame")
        self._content_window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.content.bind("<Configure>", self._update_scrollregion)
        self.canvas.bind("<Configure>", self._resize_content)

        # A toplevel binding also sees events from buttons inside the content.
        # Filter by pointer position to keep scrolling local to this container.
        self._toplevel = self.winfo_toplevel()
        self._windowing_system = str(self.tk.call("tk", "windowingsystem"))
        self._wheel_bindings = {
            sequence: self._toplevel.bind(sequence, self._on_mousewheel, add="+")
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>")
        }
        self.bind("<Destroy>", self._remove_wheel_bindings)

    def _update_scrollregion(self, _event: tk.Event | None = None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_content(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self._content_window, width=event.width)
        self._update_scrollregion()

    def _on_mousewheel(self, event: tk.Event) -> str | None:
        widget = self.winfo_containing(event.x_root, event.y_root)
        while widget is not None and widget is not self:
            widget = widget.master
        if widget is not self:
            return None

        if event.num == 4:
            units = -1
        elif event.num == 5:
            units = 1
        elif not event.delta:
            return None
        elif self._windowing_system == "aqua":
            units = -int(event.delta)
        else:
            units = -int(event.delta / 120)
            if units == 0:
                units = -1 if event.delta > 0 else 1

        # Canvas can move short content away from its starting position, so
        # leave it at the top when the whole column already fits on screen.
        if self.canvas.yview() != (0.0, 1.0):
            self.canvas.yview_scroll(units, "units")
        return "break"

    def _remove_wheel_bindings(self, event: tk.Event) -> None:
        if event.widget is not self:
            return
        for sequence, binding_id in self._wheel_bindings.items():
            try:
                self._toplevel.unbind(sequence, binding_id)
            except tk.TclError:
                # The owning toplevel may already be in the process of closing.
                pass
        self._wheel_bindings.clear()

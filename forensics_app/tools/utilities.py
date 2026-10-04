import tkinter as tk


def dialog_options(parent: tk.Misc, title: str, message: str, options: list):
    result = {"value": None}

    win = tk.Toplevel(parent)
    win.title(title)
    win.transient(parent)
    win.resizable(False, False)

    tk.Label(win, text=message).pack(padx=15, pady=15)

    frame = tk.Frame(win)
    frame.pack(padx=15, pady=(0, 15))

    def choose(operation):
        result["value"] = operation
        win.destroy()

    for op in options:
        tk.Button(frame, text=op, command=lambda o=op: choose(o)).pack(side="left", padx=5)

    win.protocol("WM_DELETE_WINDOW", win.destroy)
    win.grab_set()
    parent.wait_window(win)
    return result["value"]
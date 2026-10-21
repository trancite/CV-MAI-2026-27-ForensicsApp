"""The application's main window and event coordination."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, UnidentifiedImageError

from forensics_app.core import ImageDocument
from forensics_app.tools.base import ForensicsTool, MatchPreview
from forensics_app.tools.registry import ToolRegistry
from .image_view import ImageView
from .match_view import MatchView
from .scrollable_frame import ScrollableFrame


OPEN_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]
SAVE_TYPES = [("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("TIFF image", "*.tiff")]


class MainWindow:
    def __init__(self, root: tk.Tk, registry: ToolRegistry) -> None:
        self.root = root
        self.registry = registry
        self.document = ImageDocument()
        self.comparison_image: Image.Image | None = None
        self.comparison_path: Path | None = None
        self.comparison_title = tk.StringVar(value="Original")
        self.match_preview: MatchPreview | None = None
        self.show_matches = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Ready. Open an image to begin.")

        self._configure_window()
        self._build_menu()
        self._build_layout()
        self._bind_shortcuts()
        self._refresh()

    def _configure_window(self) -> None:
        self.root.title("ForensicsApp")
        self.root.geometry("1180x720")
        self.root.minsize(820, 520)
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Sidebar.TFrame", background="#eef1f5")
        style.configure("Category.TLabel", background="#eef1f5", font=("TkDefaultFont", 10, "bold"))
        style.configure("Tool.TButton", anchor="w", padding=(10, 7))
        style.configure("Title.TLabel", font=("TkDefaultFont", 16, "bold"))

    def _build_menu(self) -> None:
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Open image…", command=self.open_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Open comparison image…", command=self.open_comparison_image)
        file_menu.add_command(label="Save result as…", command=self.save_image, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        menu.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menu, tearoff=False)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        edit_menu.add_command(label="Reset to original", command=self.reset)
        menu.add_cascade(label="Edit", menu=edit_menu)

        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="About", command=self.show_about)
        menu.add_cascade(label="Help", menu=help_menu)
        self.root.configure(menu=menu)

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True)

        toolbar = ttk.Frame(container, padding=(10, 8))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Open image", command=self.open_image).pack(side="left")
        ttk.Button(toolbar, text="Save result", command=self.save_image).pack(side="left", padx=(6, 0))
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)
        self.undo_button = ttk.Button(toolbar, text="Undo", command=self.undo)
        self.undo_button.pack(side="left")
        self.redo_button = ttk.Button(toolbar, text="Redo", command=self.redo)
        self.redo_button.pack(side="left", padx=(6, 0))
        ttk.Button(toolbar, text="Reset", command=self.reset).pack(side="left", padx=(6, 0))
        self.matches_button = ttk.Checkbutton(
            toolbar, text="Show matches", variable=self.show_matches, command=self._refresh,
        )
        self.matches_button.pack(side="left", padx=(12, 0))

        body = ttk.Panedwindow(container, orient="horizontal")
        body.pack(fill="both", expand=True)

        sidebar = ttk.Frame(body, style="Sidebar.TFrame", padding=12, width=235)
        sidebar.pack_propagate(False)
        body.add(sidebar, weight=0)
        ttk.Label(sidebar, text="Forensics tools", style="Title.TLabel", background="#eef1f5").pack(
            anchor="w", pady=(0, 12)
        )
        self.tool_list = ScrollableFrame(sidebar)
        self.tool_list.pack(fill="both", expand=True)
        for category, tools in self.registry.categories():
            ttk.Label(self.tool_list.content, text=category, style="Category.TLabel").pack(
                anchor="w", pady=(9, 4)
            )
            for tool in tools:
                button = ttk.Button(
                    self.tool_list.content,
                    text=tool.title,
                    style="Tool.TButton",
                    command=lambda selected=tool: self.run_tool(selected),
                )
                button.pack(fill="x", pady=2)
                button.bind("<Enter>", lambda _event, selected=tool: self.status.set(selected.description))
                button.bind("<Leave>", lambda _event: self.status.set("Ready."))

        image_area = ttk.Frame(body)
        body.add(image_area, weight=1)
        comparison_actions = ttk.Frame(image_area, padding=(8, 6))
        comparison_actions.pack(fill="x")
        ttk.Button(comparison_actions, text="Open comparison…", command=self.open_comparison_image).pack(
            side="left"
        )
        self.use_original_button = ttk.Button(comparison_actions, text="Use original", command=self.use_original)
        self.use_original_button.pack(side="left", padx=(6, 0))

        comparison = ttk.Panedwindow(image_area, orient="horizontal")
        self.comparison_panes = comparison
        comparison.pack(fill="both", expand=True)
        self.match_view = MatchView(image_area)
        reference_pane = ttk.Frame(comparison)
        comparison.add(reference_pane, weight=1)
        ttk.Label(reference_pane, textvariable=self.comparison_title, width=1, padding=(8, 6)).pack(fill="x")
        self.comparison_view = ImageView(reference_pane)
        self.comparison_view.pack(fill="both", expand=True)

        result_pane = ttk.Frame(comparison)
        comparison.add(result_pane, weight=1)
        ttk.Label(result_pane, text="Current result", width=1, padding=(8, 6)).pack(fill="x")
        self.image_view = ImageView(result_pane)
        self.image_view.pack(fill="both", expand=True)

        inspector = ttk.Frame(body, padding=12, width=250)
        inspector.pack_propagate(False)
        body.add(inspector, weight=0)
        ttk.Label(inspector, text="Results", style="Title.TLabel").pack(anchor="w", pady=(0, 10))
        self.results = ttk.Treeview(inspector, columns=("value",), show="tree headings", height=15)
        self.results.heading("#0", text="Property")
        self.results.heading("value", text="Value")
        self.results.column("#0", width=95, stretch=True)
        self.results.column("value", width=120, stretch=True)
        self.results.pack(fill="both", expand=True)

        ttk.Label(container, textvariable=self.status, anchor="w", padding=(10, 6), relief="sunken").pack(fill="x")

    def _bind_shortcuts(self) -> None:
        self.root.bind_all("<Control-o>", lambda _event: self.open_image())
        self.root.bind_all("<Control-Shift-S>", lambda _event: self.save_image())
        self.root.bind_all("<Control-z>", lambda _event: self.undo())
        self.root.bind_all("<Control-y>", lambda _event: self.redo())

    def open_image(self) -> None:
        filename = filedialog.askopenfilename(title="Open evidence image", filetypes=OPEN_TYPES)
        if not filename:
            return
        try:
            self.document.load(filename)
        except (OSError, UnidentifiedImageError) as error:
            messagebox.showerror("Could not open image", str(error), parent=self.root)
            return
        self.status.set(f"Opened {Path(filename).name}")
        self._clear_matches()
        self._show_default_details()
        self._refresh()

    def open_comparison_image(self) -> None:
        filename = filedialog.askopenfilename(title="Open comparison image", filetypes=OPEN_TYPES)
        if not filename:
            return
        try:
            with Image.open(filename) as image:
                comparison_image = image.copy()
        except (OSError, UnidentifiedImageError) as error:
            messagebox.showerror("Could not open comparison image", str(error), parent=self.root)
            return
        self.comparison_image = comparison_image
        self.comparison_path = Path(filename)
        self._clear_matches()
        self.status.set(f"Comparing with {self.comparison_path.name}. Tools apply to the current result.")
        self._refresh()

    def use_original(self) -> None:
        self._clear_matches()
        self.comparison_image = None
        self.comparison_path = None
        self.status.set("Showing the original image beside the current result.")
        self._refresh()

    def save_image(self) -> None:
        if not self._require_image():
            return
        source = self.document.path
        match_map = self.match_preview if self.show_matches.get() else None
        suffix = "matches" if match_map is not None else "result"
        initial = f"{source.stem}_{suffix}.png" if source else f"{suffix}.png"
        filename = filedialog.asksaveasfilename(
            title="Save match map" if match_map is not None else "Save processed image",
            defaultextension=".png",
            initialfile=initial,
            filetypes=SAVE_TYPES,
        )
        if not filename:
            return
        try:
            if match_map is not None:
                match_map.export_image.save(filename)
            else:
                self.document.save(filename)
        except OSError as error:
            messagebox.showerror("Could not save image", str(error), parent=self.root)
            return
        self.status.set(f"Saved result as {Path(filename).name}")

    def run_tool(self, tool: ForensicsTool) -> None:
        if tool.requires_image and not self._require_image():
            return
        try:
            result = tool.run(self.root, self.document)
        except Exception as error:  # keep one student feature from crashing the shell
            messagebox.showerror(f"{tool.title} failed", str(error), parent=self.root)
            self.status.set(f"Error in {tool.title}.")
            return
        if result is None:
            self.status.set(f"Cancelled {tool.title}.")
            return
        self._clear_matches()
        if result.image is not None:
            self.document.apply(result.image)
        if result.match_preview is not None:
            self.match_preview = result.match_preview
            self.comparison_image = result.match_preview.left_image
            self.comparison_path = result.match_preview.reference_path
            self.show_matches.set(True)
        self._show_details(result.details)
        self.status.set(result.message)
        self._refresh()

    def undo(self) -> None:
        if self.document.undo():
            self._clear_matches()
            self.status.set("Undid the last image operation.")
            self._show_default_details()
            self._refresh()

    def redo(self) -> None:
        if self.document.redo():
            self._clear_matches()
            self.status.set("Redid the image operation.")
            self._show_default_details()
            self._refresh()

    def reset(self) -> None:
        if self.document.reset():
            self._clear_matches()
            self.status.set("Restored the original image.")
            self._show_default_details()
            self._refresh()

    def _require_image(self) -> bool:
        if self.document.is_loaded:
            return True
        messagebox.showinfo("No image loaded", "Open an image first.", parent=self.root)
        return False

    def _clear_matches(self) -> None:
        self.match_preview = None
        self.show_matches.set(False)

    def _refresh(self) -> None:
        reference = self.comparison_image if self.comparison_image is not None else self.document.original
        self.comparison_view.show(reference)
        self.comparison_title.set(f"Comparison: {self.comparison_path.name}" if self.comparison_path else "Original")
        self.use_original_button.configure(state="normal" if self.comparison_image is not None else "disabled")
        self.image_view.show(self.document.current)
        self.matches_button.configure(state="normal" if self.match_preview is not None else "disabled")
        if self.show_matches.get() and self.match_preview is not None:
            self.comparison_panes.pack_forget()
            self.match_view.pack(fill="both", expand=True)
            self.match_view.show(self.match_preview)
        else:
            self.match_view.pack_forget()
            self.match_view.show(None)
            self.comparison_panes.pack(fill="both", expand=True)
        self.undo_button.configure(state="normal" if self.document.can_undo else "disabled")
        self.redo_button.configure(state="normal" if self.document.can_redo else "disabled")
        title = self.document.path.name if self.document.path else "No image"
        self.root.title(f"ForensicsApp — {title}")

    def _show_default_details(self) -> None:
        image = self.document.current
        if image is None:
            self._show_details({})
            return
        self._show_details(
            {
                "File": self.document.path.name if self.document.path else "—",
                "Size": f"{image.width} × {image.height}",
                "Mode": image.mode,
            }
        )

    def _show_details(self, details: dict[str, object]) -> None:
        for item in self.results.get_children():
            self.results.delete(item)
        for name, value in details.items():
            self.results.insert("", "end", text=str(name), values=(str(value),))

    def show_about(self) -> None:
        messagebox.showinfo(
            "About ForensicsApp",
            "A modular image-forensics application for the Computer Vision course.\n\n"
            "Add each weekly feature as a tool in forensics_app/tools/.",
            parent=self.root,
        )

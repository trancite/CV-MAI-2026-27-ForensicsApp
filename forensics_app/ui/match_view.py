"""Resizable images and connecting lines drawn together on a Tk canvas."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from forensics_app.tools.base import MatchPreview


class MatchView(ttk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, padding=8)
        self.canvas = tk.Canvas(self, background="#20242b", borderwidth=0, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self.bind("<Destroy>", self._on_destroy)
        self._preview: MatchPreview | None = None
        self._photos: list[ImageTk.PhotoImage] = []
        self._resize_job: str | None = None

    def show(self, preview: MatchPreview | None) -> None:
        self._preview = preview
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
            self._resize_job = None
        self._render()

    def _on_resize(self, _event: tk.Event) -> None:
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(50, self._render)

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is self and self._resize_job is not None:
            self.after_cancel(self._resize_job)
            self._resize_job = None

    def _render(self) -> None:
        self._resize_job = None
        self.canvas.delete("all")
        self._photos.clear()
        if self._preview is None:
            return
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        half_width = width / 2
        available = (max(int(half_width) - 24, 1), max(height - 64, 1))
        transforms = []
        images = (self._preview.left_image, self._preview.right_image)
        titles = (f"Reference: {self._preview.reference_path.name}", "Current image")
        for index, (source, title) in enumerate(zip(images, titles)):
            thumbnail = source.copy()
            thumbnail.thumbnail(available, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(thumbnail)
            self._photos.append(photo)
            center_x = half_width * (index + 0.5)
            offset_x = center_x - thumbnail.width / 2
            offset_y = 40 + (max(height - 40, 1) - thumbnail.height) / 2
            self.canvas.create_text(
                center_x, 18, text=title, fill="#c7ccd4", width=max(int(half_width) - 16, 1),
            )
            self.canvas.create_image(offset_x, offset_y, image=photo, anchor="nw", tags="image")
            transforms.append((offset_x, offset_y, thumbnail.width / source.width, thumbnail.height / source.height))

        for left_point, right_point in self._preview.point_pairs:
            endpoints = []
            for (x, y), (offset_x, offset_y, scale_x, scale_y) in zip((left_point, right_point), transforms):
                endpoints.append((offset_x + x * scale_x, offset_y + y * scale_y))
            self.canvas.create_line(*endpoints[0], *endpoints[1], fill="yellow", width=1, tags="match")
            for x, y in endpoints:
                self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, outline="red", tags="keypoint")

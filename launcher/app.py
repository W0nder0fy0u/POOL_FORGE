"""PoolForge desktop app: paste the join link, click Join, keep the window open.

    PoolForge.exe                                  # asks for the link
    PoolForge.exe https://my-pool.onrender.com/join/X7K2AB   # joins right away

Run from source with `python -m launcher.app` (uses the repo's agent code).
Setup work runs in a background thread; it talks to the window only through a queue.
"""

import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from launcher import setup_env as env


class App:
    def __init__(self, root: tk.Tk, link: str = ""):
        self.root = root
        self.events: "queue.Queue[tuple]" = queue.Queue()
        self.proc = None
        self.leaving = False
        root.title("PoolForge")
        root.minsize(560, 440)
        root.protocol("WM_DELETE_WINDOW", self.close)

        frame = ttk.Frame(root, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="PoolForge", font=("TkDefaultFont", 16, "bold")).pack(anchor="w")
        ttk.Label(frame, text="Share this laptop's spare RAM and compute with your friends' pool.").pack(anchor="w", pady=(0, 12))

        ttk.Label(frame, text="Join link (or code) your friend sent you:").pack(anchor="w")
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(2, 10))
        self.link = tk.StringVar(value=link or env.load_config().get("last_link", ""))
        self.entry = ttk.Entry(row, textvariable=self.link)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda _e: self.join())
        self.join_btn = ttk.Button(row, text="Join", command=self.join)
        self.join_btn.pack(side="left", padx=(8, 0))
        self.leave_btn = ttk.Button(row, text="Leave", command=self.leave, state="disabled")
        self.leave_btn.pack(side="left", padx=(8, 0))

        # Unticked: the CPU build of PyTorch (about 200 MB) and --device cpu, so the GPU is never used.
        has_gpu = env.has_nvidia_gpu()
        self.share_gpu = tk.BooleanVar(value=has_gpu and env.load_config().get("share_gpu", True))
        self.gpu_box = ttk.Checkbutton(frame, variable=self.share_gpu, state="normal" if has_gpu else "disabled",
                                       text="Share my NVIDIA GPU (the first time this downloads about 2.5 GB)" if has_gpu
                                       else "Share my GPU (no NVIDIA GPU found: this laptop shares its CPU)")
        self.gpu_box.pack(anchor="w", pady=(0, 10))
        self.gpu_box_state = "normal" if has_gpu else "disabled"

        self.status = tk.StringVar(value="Paste the link and click Join.")
        status = ttk.Label(frame, textvariable=self.status, font=("TkDefaultFont", 10, "bold"), wraplength=520)
        status.pack(anchor="w", fill="x")
        status.bind("<Configure>", lambda e: status.configure(wraplength=max(e.width, 200)))
        self.bar = ttk.Progressbar(frame, mode="determinate", maximum=1.0)
        self.bar.pack(fill="x", pady=(6, 10))

        self.log_box = ScrolledText(frame, height=12, state="disabled", wrap="word", font=("TkFixedFont", 9))
        self.log_box.pack(fill="both", expand=True)
        ttk.Label(frame, text=f"PoolForge {env.VERSION} · files in {env.data_dir()}", foreground="gray").pack(anchor="w", pady=(6, 0))

        root.after(100, self.poll)
        if link:
            root.after(200, self.join)

    # ---------------------------------------------------------------- window side
    def poll(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "log":
                    self.log_box.configure(state="normal")
                    self.log_box.insert("end", value + "\n")
                    self.log_box.see("end")
                    self.log_box.configure(state="disabled")
                elif kind == "status":
                    self.status.set(value)
                elif kind == "progress":
                    self.set_progress(value)
                elif kind == "busy":
                    self.set_busy(value)
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def set_progress(self, value) -> None:
        if value is None:
            if str(self.bar["mode"]) != "indeterminate":
                self.bar.configure(mode="indeterminate")
                self.bar.start(15)
        else:
            if str(self.bar["mode"]) != "determinate":
                self.bar.stop()
                self.bar.configure(mode="determinate")
            self.bar["value"] = value

    def set_busy(self, busy: bool) -> None:
        self.join_btn.configure(state="disabled" if busy else "normal")
        self.entry.configure(state="disabled" if busy else "normal")
        self.leave_btn.configure(state="normal" if busy else "disabled")
        self.gpu_box.configure(state="disabled" if busy else self.gpu_box_state)
        if not busy:
            self.set_progress(0)

    def join(self) -> None:
        if str(self.join_btn["state"]) == "disabled":
            return
        try:
            server, code = env.parse_link(self.link.get(), env.load_config().get("server") or env.DEFAULT_SERVER)
        except ValueError as exc:
            self.status.set(str(exc))
            return
        use_gpu = self.share_gpu.get()
        if self.gpu_box_state == "normal":  # keep the saved choice when there is no GPU to choose about
            env.save_config(share_gpu=use_gpu)
        env.save_config(last_link=self.link.get().strip(), server=server)
        self.leaving = False
        self.set_busy(True)
        threading.Thread(target=self.run, args=(server, code, use_gpu), daemon=True).start()

    def leave(self) -> None:
        self.leaving = True
        self.status.set("Leaving the pool...")
        if self.proc:
            threading.Thread(target=env.stop_agent, args=(self.proc,), daemon=True).start()

    def close(self) -> None:
        if self.proc:
            env.stop_agent(self.proc)
        self.root.destroy()

    # ---------------------------------------------------------------- worker side
    def send(self, kind: str, value=None) -> None:
        self.events.put((kind, value))

    def run(self, server: str, code: str, use_gpu: bool) -> None:
        try:
            self.send("status", "Setting up (the first time takes a few minutes)...")
            lib = env.prepare(lambda m: self.send("log", m), lambda p: self.send("progress", p), use_gpu)
            if self.leaving:
                return
            self.send("progress", None)
            self.send("status", f"Connecting to pool {code}...")
            self.proc = env.start_agent(server, code, lib, use_gpu)
            for line in self.proc.stdout:
                line = line.rstrip()
                self.send("log", line)
                if "Connected to coordinator" in line:
                    self.send("progress", 1.0)
                    self.send("status", f"Connected to pool {code}. Keep this window open; close it to leave.")
                elif "Cannot reach" in line or "Lost coordinator connection" in line:
                    self.send("progress", None)
                    self.send("status", "Can't reach the pool server. Retrying...")
            exit_code = self.proc.wait()
            self.send("status", "Left the pool." if self.leaving
                      else f"The agent stopped (exit code {exit_code}). See the messages above, then click Join to retry.")
        except env.SetupError as exc:
            self.send("status", str(exc))
        except Exception as exc:  # show anything unexpected instead of dying silently
            self.send("log", repr(exc))
            self.send("status", f"Something went wrong: {exc}")
        finally:
            self.proc = None
            self.send("busy", False)


def main() -> None:
    root = tk.Tk()
    App(root, link=sys.argv[1] if len(sys.argv) > 1 else "")
    if os.environ.get("POOLFORGE_SELFTEST") == "1":
        # CI builds the binary then runs it like this to prove the window (Tk) really loads
        # and the compiled agent was bundled (__compiled__ only exists in the Nuitka build).
        if "__compiled__" in globals() and not env.has_agent(env.PAYLOAD_DIR):
            sys.exit(f"Self-test failed: no compiled agent in {env.PAYLOAD_DIR}")
        root.update()
        root.after(300, root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()

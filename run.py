# run.py

import sys
from pathlib import Path

# Adicione a pasta src ao sys.path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

import tkinter as tk
from gui import VibratoGUI
from splash import show_splash

def main():
    root = tk.Tk()
    show_splash(root)
    app = VibratoGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

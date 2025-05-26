# run.py

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

#!/usr/bin/env python3
from leap_hand_gui import LeapHandGUI
import tkinter as tk

def main():
    root = tk.Tk()
    gui = LeapHandGUI(root)
    root.protocol("WM_DELETE_WINDOW", gui.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()

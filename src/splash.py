# splash.py

import tkinter as tk
from PIL import Image, ImageTk
from tkinter import font as tkFont

def show_splash(root):
    splash = tk.Toplevel(root)
    splash.title("About Vibrato Scope")
    splash.geometry("800x600")
    splash.configure(bg="#b0afa6")
    splash.resizable(False, False)

    # Fontes
    cal_font = tkFont.Font(family="Calibri", size=10)
    cal_title = tkFont.Font(family="Calibri", size=14)

    # Logo no canto superior esquerdo
    try:
        img = Image.open("logo.png").resize((150, 150))
        logo = ImageTk.PhotoImage(img)
        logo_label = tk.Label(splash, image=logo, bg="#b0afa6", borderwidth=0, highlightthickness=0)
        logo_label.image = logo
        logo_label.place(x=10, y=10)
    except Exception:
        logo_label = tk.Label(splash, text="[Logo not found]", font=cal_title, bg="#b0afa6", fg="white")
        logo_label.place(x=20, y=20)

    # Texto com fundo combinando com a janela
    splash_text = (
        "Developed by: Dr. Tiago Lima Bicalho Cruz\n\n"
        "Ph.D. in Music (2020–2024)\n"
        "Federal University of Minas Gerais (UFMG)\n\n"
        "Additional Qualifications:\n"
        "M.A. in Music, Postgraduate Specialization in Voice Clinics,\n"
        "B.Sc. in Speech-Language Therapy and Audiology\n\n"
        "Teaching & Research:\n"
        "Vocal Coach, Lecturer, and Researcher in voice and acoustic analysis\n\n"
        "Disclaimer: This software is for personal use only. It is provided as-is without any warranty.\n\n"
        "License: Although this software is open source under the MIT license, the author kindly requests that users\n"
        "do not modify or redistribute altered versions.\n\n"
        "If you have suggestions, please send an e-mail.\n"
        "Use the tool as-is and cite it appropriately. Thank you!\n\n"
        "The analysis results are approximate and may not be 100% accurate.\n\n"
        "© 2025 Dr. Tiago Lima Bicalho Cruz\n"
        "tiagolbc@gmail.com\n\n"
        "Press 'Continue' to proceed."
    )

    label = tk.Label(
        splash,
        text=splash_text,
        font=cal_font,
        justify="center",
        wraplength=700,
        fg="black",
        bg="#b0afa6",  # Match window background
        borderwidth=0,
        highlightthickness=0
    )
    label.place(relx=0.5, rely=0.55, anchor="center")  # Slightly lower to avoid logo

    # Botão continue centralizado na parte inferior
    btn = tk.Button(
        splash,
        text="Continue",
        command=splash.destroy,
        font=cal_title,
        relief="raised",
        borderwidth=3,
        padx=8,
        pady=4
    )
    btn.pack(side="bottom", pady=(0, 10))

    # Centraliza a janela na tela
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (800 // 2)
    y = (root.winfo_screenheight() // 2) - (600 // 2)
    splash.geometry(f"+{x}+{y}")

    # Forçar renderização
    splash.update_idletasks()
    splash.after(500, splash.update)

    splash.grab_set()
    root.wait_window(splash)
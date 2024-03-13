import tkinter as tk
from tkinter import filedialog

from localscribe.skeleton import transcribe_audio


def browse_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        transcribe_audio(file_path)


def scribe():
    root = tk.Tk()
    root.title("Audio Transcription App")

    browse_button = tk.Button(root, text="Browse", command=browse_file)
    browse_button.pack(pady=20)

    root.mainloop()


if __name__ == "__main__":
    scribe()

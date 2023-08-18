import tkinter as tk
from tkinter import filedialog
import os
import subprocess

class Application:
    def __init__(self, master):
        self.master = master
        master.title("Face Recognition Interface")
        master.geometry("500x400")

        # Pastel color scheme
        self.bg_color = "#F9DCD1"
        self.button_color = "#E0BBE4"
        self.label_color = "#E7CECB"

        self.master.configure(bg=self.bg_color)

        self.label = tk.Label(master, text="Select an option:", bg=self.label_color, font=("Arial", 20))
        self.label.pack(pady=20)

        self.capture_name_label = tk.Label(master, text="Enter name:", bg=self.bg_color, font=("Arial", 14))
        self.capture_name_label.pack()

        self.capture_name_entry = tk.Entry(master, font=("Arial", 14))
        self.capture_name_entry.pack()

        self.capture_roll_label = tk.Label(master, text="Enter roll no:", bg=self.bg_color, font=("Arial", 14))
        self.capture_roll_label.pack()

        self.capture_roll_entry = tk.Entry(master, font=("Arial", 14))
        self.capture_roll_entry.pack()

        self.capture_button = tk.Button(master, text="Capture Images", command=self.capture_images, bg=self.button_color, font=("Arial", 16), height=2)
        self.capture_button.pack(pady=20)

        self.train_button = tk.Button(master, text="Train Model", command=self.train_model, bg=self.button_color, font=("Arial", 16), height=2)
        self.train_button.pack(pady=20)

        self.recognize_button = tk.Button(master, text="Recognize Faces", command=self.recognize_faces, bg=self.button_color, font=("Arial", 16), height=2)
        self.recognize_button.pack(pady=20)

        self.attendance_button = tk.Button(master, text="Attendance sheet", command=self.open_attendance_sheet, bg=self.button_color, font=("Arial", 16), height=2)
        self.attendance_button.pack(pady=20)

    def capture_images(self):
        name = self.capture_name_entry.get()
        roll_no = self.capture_roll_entry.get()

        subprocess.call(['python', 'images.py', name, roll_no])

    def train_model(self):
        subprocess.call(['python', 'train_model.py'])

    def recognize_faces(self):
        subprocess.call(['python', 'recognize_faces.py'])


    def open_attendance_sheet(self):
        os.startfile('attendance.csv')
        
root = tk.Tk()
app = Application(root)
root.mainloop()

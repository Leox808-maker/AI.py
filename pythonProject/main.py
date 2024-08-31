import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import EmotionRecognizer
import threading

class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Analysis Suite")
        self.root.geometry("500x400")
        self.root.configure(bg="#f0f0f0")
        self.recognizer = EmotionRecognizer()

        self.create_widgets()
    def create_widgets(self):
        title_label = ttk.Label(self.root, text="Face Analysis Suite", font=("Helvetica", 18), background="#f0f0f0")
        title_label.pack(pady=20)

        description_label = ttk.Label(self.root, text="Seleziona una funzionalità per iniziare:", font=("Helvetica", 12), background="#f0f0f0")
        description_label.pack(pady=10)

        self.create_buttons()
        self.create_status_bar()
    def create_buttons(self):
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=20)
        emotion_button = ttk.Button(button_frame, text="Riconoscimento Emozioni", command=self.recognize_emotions, width=30)
        emotion_button.grid(row=0, column=0, padx=10, pady=10)
        blink_button = ttk.Button(button_frame, text="Riconoscimento Blink Occhi", command=self.recognize_blink, width=30)
        blink_button.grid(row=0, column=1, padx=10, pady=10)
        age_button = ttk.Button(button_frame, text="Riconoscimento Età", command=self.recognize_age, width=30)
        age_button.grid(row=1, column=0, padx=10, pady=10)

        eye_tracking_button = ttk.Button(button_frame, text="Eye Tracking", command=self.eye_tracking, width=30)
        eye_tracking_button.grid(row=1, column=1, padx=10, pady=10)
        settings_button = ttk.Button(button_frame, text="Impostazioni", command=self.settings, width=30)
        settings_button.grid(row=2, column=0, columnspan=2, pady=20)

    def create_status_bar(self):
        self.status_var = tk.StringVar()
        self.status_var.set("Pronto")

        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, ipady=2)
    def update_status(self, message):
        self.status_var.set(message)
    def recognize_emotions(self):
        self.update_status("Riconoscimento delle emozioni in corso...")
        messagebox.showinfo("Riconoscimento Emozioni", "Funzione non ancora implementata.")
        self.update_status("Pronto")
        #recognition_thread = Thread(target=self.recognizer.start_recognition)
        #recognition_thread.start()
        self.update_status("Pronto")

    def recognize_blink(self):
        self.update_status("Riconoscimento blink degli occhi in corso...")
        messagebox.showinfo("Riconoscimento Blink Occhi", "Funzione non ancora implementata.")
        self.update_status("Pronto")
    def recognize_age(self):
        self.update_status("Riconoscimento dell'età in corso...")
        messagebox.showinfo("Riconoscimento Età", "Funzione non ancora implementata.")
        self.update_status("Pronto")
    def eye_tracking(self):
        self.update_status("Eye tracking in corso...")
        messagebox.showinfo("Eye Tracking", "Funzione non ancora implementata.")
        self.update_status("Pronto")
    def settings(self):
        self.update_status("Apertura impostazioni...")
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Impostazioni")
        settings_window.geometry("400x300")
        settings_window.configure(bg="#f0f0f0")

        volume_label = ttk.Label(settings_window, text="Volume", font=("Helvetica", 12), background="#f0f0f0")
        volume_label.pack(pady=10)

        volume_slider = ttk.Scale(settings_window, from_=0, to=100, orient=tk.HORIZONTAL)
        volume_slider.pack(pady=10)

        brightness_label = ttk.Label(settings_window, text="Luminosità", font=("Helvetica", 12), background="#f0f0f0")
        brightness_label.pack(pady=10)

        brightness_slider = ttk.Scale(settings_window, from_=0, to=100, orient=tk.HORIZONTAL)
        brightness_slider.pack(pady=10)

        save_button = ttk.Button(settings_window, text="Salva", command=lambda: self.save_settings(settings_window))
        save_button.pack(pady=20)

        self.update_status("Pronto")

    def save_settings(self, window):
        messagebox.showinfo("Impostazioni", "Impostazioni salvate con successo!")
        window.destroy()
        self.update_status("Pronto")

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()


    def on_button_click(self, button):
        if button == "Riconoscimento emozioni":
            self.start_emotion_recognition()
        elif button == "Riconoscimento blink occhi":
            self.start_blink_recognition()
        elif button == "Riconoscimento età":
            self.start_age_recognition()
        elif button == "Eye tracking":
            self.start_eye_tracking()
        elif button == "Impostazioni":
            self.open_settings()
        else:
            self.show_message(f"La funzione '{button}' non è ancora stata implementata.")

    def show_message(self, message):
        messagebox.showinfo("Info", message)
    def start_emotion_recognition(self):
        self.show_message("La funzione di riconoscimento delle emozioni non è ancora stata implementata.")
    def start_blink_recognition(self):
        self.show_message("La funzione di riconoscimento del blink occhi non è ancora stata implementata.")
    def start_age_recognition(self):
        self.show_message("La funzione di riconoscimento dell'età non è ancora stata implementata.")
    def start_eye_tracking(self):
        self.show_message("La funzione di eye tracking non è ancora stata implementata.")
    def open_settings(self):
        self.show_message("La funzione di impostazioni non è ancora stata implementata.")


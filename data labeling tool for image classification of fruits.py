#kidus_berhanu
import os
import cv2
from tkinter import *

class DataLabeler:
    def __init__(self):
        self.root = Tk()
        self.root.title("Fruit Image Classifier")
        self.root.geometry("400x200")

        self.label = Label(self.root, text="Select a fruit:")
        self.label.pack()

        self.var = StringVar(self.root)
        self.var.set("Apple")

        self.dropdown = OptionMenu(self.root, self.var, "Apple", "Banana", "Orange", "Mango")
        self.dropdown.pack()

        self.image_label = Label(self.root)
        self.image_label.pack()

        self.load_button = Button(self.root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.save_button = Button(self.root, text="Save Label", command=self.save_label)
        self.save_button.pack()

        self.edit_button = Button(self.root, text="Edit Label", command=self.edit_label)
        self.edit_button.pack()

        self.current_file_path = ""
        self.labels = {}
        self.read_labels_from_file()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path.split(".")[-1] not in ["jpg", "jpeg", "png"]:
            messagebox.showerror("Error", "Invalid file format. Only jpg, jpeg and png are supported.")
            return

        self.current_file_path = file_path
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img

        if file_path in self.labels:
            self.var.set(self.labels[file_path])

    def save_label(self):
        if self.current_file_path == "":
            messagebox.showerror("Error", "No image selected. Please load an image first.")
            return

        fruit = self.var.get()
        self.labels[self.current_file_path] = fruit
        self.write_labels_to_file()

    def edit_label(self):
        if self.current_file_path == "":
            messagebox.showerror("Error", "No image selected. Please load an image first.")
            return

        self.var.set(self.labels[self.current_file_path])

    def read_labels_from_file(self):
        if not
os.path.exists("labels.txt"):
return
with open("labels.txt", "r") as f:
lines = f.readlines()
for line in lines:
file_path, label = line.strip().split(",")
self.labels[file_path] = label
def write_labels_to_file(self):
    with open("labels.txt", "w") as f:
        for file_path, label in self.labels.items():
            f.write(f"{file_path},{label}\n")

def run(self):
    self.root.mainloop()
if name == "main":
labeler = DataLabeler()
labeler.run()

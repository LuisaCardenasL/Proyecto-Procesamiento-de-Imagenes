import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import nibabel as nib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.filters import threshold_isodata

import matplotlib.pyplot as plt

class MedicalImageGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Visualizador de Imágenes Médicas")

        self.image = None
        self.ax = None
        self.canvas = None
        self.selected_dimension = None
        self.layer_scale = None
        self.segmented_image = None

        self.create_menu()
        self.create_widgets()

    def create_menu(self):
        menubar = tk.Menu(self.master)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Abrir", command=self.load_image)
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self.master.quit)
        menubar.add_cascade(label="Archivo", menu=file_menu)

        segment_menu = tk.Menu(menubar, tearoff=0)
        segment_menu.add_command(label="Segmentar por Umbralización", command=self.thresholding)
        segment_menu.add_command(label="Segmentar por ISODATA", command=self.segmentation_isodata)
        menubar.add_cascade(label="Segmentar", menu=segment_menu)

        self.master.config(menu=menubar)

    def create_widgets(self):
        self.load_button = tk.Button(self.master, text="Cargar Archivo", command=self.load_image)
        self.load_button.pack(side=tk.TOP)

        self.dimension_label = tk.Label(self.master, text="Seleccionar Dimensión:")
        self.dimension_label.pack(side=tk.TOP)

        self.dimension_combobox = ttk.Combobox(self.master, state="disabled")
        self.dimension_combobox.pack(side=tk.TOP)

        self.layer_label = tk.Label(self.master, text="Capa:")
        self.layer_label.pack(side=tk.TOP)

        self.layer_scale = tk.Scale(self.master, orient=tk.HORIZONTAL, command=self.display_image)
        self.layer_scale.pack(side=tk.TOP, fill=tk.X)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii")])
        if file_path:
            self.image = nib.load(file_path)
            self.populate_dimensions_combobox()

    def populate_dimensions_combobox(self):
        dimensions = ['X', 'Y', 'Z']
        self.dimension_combobox.config(values=dimensions, state="readonly")
        self.dimension_combobox.current(0)
        self.selected_dimension = dimensions[0]
        self.dimension_combobox.bind("<<ComboboxSelected>>", self.update_selected_dimension)
        self.update_layer_scale()

    def update_selected_dimension(self, event):
        self.selected_dimension = self.dimension_combobox.get()
        self.update_layer_scale()

    def update_layer_scale(self):
        data = self.image.get_fdata()
        if self.selected_dimension == 'X':
            num_layers = data.shape[0]
        elif self.selected_dimension == 'Y':
            num_layers = data.shape[1]
        else:  # 'Z'
            num_layers = data.shape[2]

        self.layer_scale.config(from_=0, to=num_layers - 1)

    def display_image(self, event=None):
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()

        data = self.image.get_fdata()
        layer = int(self.layer_scale.get())

        if self.selected_dimension == 'X':
            image_slice = data[layer, :, :]
        elif self.selected_dimension == 'Y':
            image_slice = data[:, layer, :]
        else:  # 'Z'
            image_slice = data[:, :, layer]

        self.fig = Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(image_slice, cmap='gray')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

    def thresholding(self):
        if self.image is not None:
            # Obtener la rebanada actual según la dimensión seleccionada
            layer = int(self.layer_scale.get())
            data = self.image.get_fdata()
            if self.selected_dimension == 'X':
                image_slice = data[layer, :, :]
            elif self.selected_dimension == 'Y':
                image_slice = data[:, layer, :]
            else:  # 'Z'
                image_slice = data[:, :, layer]

            # Definir el valor inicial del umbral (tau)
            tau = 127
            # Definir el valor de la tolerancia (Delta_tau)
            delta_tau = 1
            while True:
                # Aplicar el umbral actual
                segmented_image = (image_slice > tau).astype(int)
                # Calcular la media del píxel en el primer plano (foreground) y el fondo (background)
                mean_foreground = np.mean(image_slice[segmented_image == 1])
                mean_background = np.mean(image_slice[segmented_image == 0])
                # Calcular el nuevo umbral
                new_tau = 0.5 * (mean_foreground + mean_background)
                # Verificar si la diferencia entre el nuevo umbral y el anterior es menor que la tolerancia
                if abs(new_tau - tau) < delta_tau:
                    break
                # Actualizar el umbral
                tau = new_tau
            # Convertir la imagen binaria a uint8 (0 y 255)
            segmented_image = (segmented_image * 255).astype(np.uint8)
            
            self.segmented_image = segmented_image
            self.show_segmented_image()

    def segmentation_isodata(self):
        if self.image is not None:
            # Obtener la rebanada actual según la dimensión seleccionada
            layer = int(self.layer_scale.get())
            data = self.image.get_fdata()
            if self.selected_dimension == 'X':
                image_slice = data[layer, :, :]
            elif self.selected_dimension == 'Y':
                image_slice = data[:, layer, :]
            else:  # 'Z'
                image_slice = data[:, :, layer]

            # Calcular el umbral ISODATA
            tau = threshold_isodata(image_slice)
            
            # Segmentar la imagen
            segmented_image = (image_slice > tau).astype(np.uint8) * 255
            
            self.segmented_image = segmented_image
            self.show_segmented_image()

    def show_segmented_image(self):
        if self.segmented_image is not None:
            plt.imshow(self.segmented_image, cmap='gray')
            plt.axis('off')
            plt.show()


def main():
    root = tk.Tk()
    app = MedicalImageGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

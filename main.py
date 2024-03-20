import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import nibabel as nib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.filters import threshold_isodata
from sklearn.cluster import KMeans
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

        # Annotation variables
        self.annotation_active = False
        self.circles = {}
        self.active_circle = None
        self.current_color = 'red'  # Default color
        self.brush_size = 5  # Default brush size

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
        segment_menu.add_command(label="Segmentar por K-medias", command=self.segmentation_kmeans)
        segment_menu.add_command(label="Segmentar por Crecimiento de Regiones", command=self.region_growing)  # Agregar opción de segmentación por crecimiento de regiones
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

        # Add annotation button
        self.annotation_button_text = tk.StringVar()
        self.annotation_button_text.set("Activar Anotación")
        self.annotation_button = tk.Button(self.master, textvariable=self.annotation_button_text, command=self.toggle_annotation)
        self.annotation_button.pack(side=tk.TOP)

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

    def toggle_annotation(self):
        if not self.annotation_active:
            self.activate_annotation()
            self.annotation_button_text.set("Desactivar Anotación")
        else:
            self.deactivate_annotation()
            self.annotation_button_text.set("Activar Anotación")

    def activate_annotation(self):
        if self.canvas:
            # Connect annotation functions to canvas events
            self.canvas.mpl_connect("button_press_event", self.on_click)
            self.canvas.mpl_connect("motion_notify_event", self.on_drag)
            self.canvas.mpl_connect("button_release_event", self.on_release)
        self.annotation_active = True

    def deactivate_annotation(self):
        if self.canvas:
            # Disconnect annotation functions from canvas events
            self.canvas.mpl_disconnect("button_press_event")
            self.canvas.mpl_disconnect("motion_notify_event")
            self.canvas.mpl_disconnect("button_release_event")
        self.annotation_active = False

    def on_click(self, event):
        if event.inaxes:
            x = int(event.xdata)
            y = int(event.ydata)
            color = self.current_color
            brush_size = self.brush_size

            if color in self.circles:
                if self.circles[color] in self.ax.patches:
                    self.circles[color].remove()

            circle = plt.Circle((x, y), brush_size, color=color, fill=True)
            self.ax.add_patch(circle)
            self.circles[color] = circle
            self.active_circle = circle
            self.canvas.draw()

    def on_drag(self, event):
        if event.inaxes:
            x = int(event.xdata)
            y = int(event.ydata)

            circle = self.active_circle

            if circle:
                circle.center = x, y
                self.canvas.draw()

    def on_release(self, event):
        self.active_circle = None

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

    def segmentation_kmeans(self):
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

            # Normalizar la imagen para que los valores estén entre 0 y 1
            normalized_image = image_slice / np.max(image_slice)

            # Convertir la imagen a un vector unidimensional
            flattened_image = normalized_image.flatten()

            # Aplicar K-medias con k=2
            kmeans = KMeans(n_clusters=2, random_state=0).fit(flattened_image.reshape(-1, 1))

            # Obtener las etiquetas de los clústeres
            labels = kmeans.labels_

            # Segmentar la imagen según las etiquetas obtenidas
            segmented_image = labels.reshape(normalized_image.shape) * 255

            self.segmented_image = segmented_image.astype(np.uint8)
            self.show_segmented_image()

    def region_growing(self):
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

            # Elegir un punto de inicio (semilla) para el crecimiento de regiones
            seed = (image_slice.shape[0] // 2, image_slice.shape[1] // 2)

            # Lista para almacenar las coordenadas de los píxeles pertenecientes a la región
            region = [seed]
            # Lista para almacenar los píxeles ya visitados
            visited = []

            # Umbral para la comparación de intensidades
            threshold = 20

            while region:
                # Tomar el primer píxel de la región
                x, y = region.pop(0)

                # Verificar si el píxel ya ha sido visitado
                if (x, y) in visited:
                    continue

                # Marcar el píxel como visitado
                visited.append((x, y))

                # Verificar la intensidad del píxel con sus vecinos
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        # Coordenadas del vecino
                        neighbor_x = x + i
                        neighbor_y = y + j

                        # Verificar si el vecino está dentro de la imagen
                        if 0 <= neighbor_x < image_slice.shape[0] and 0 <= neighbor_y < image_slice.shape[1]:
                            # Verificar si el vecino no ha sido visitado y su intensidad es similar al punto de inicio
                            if (neighbor_x, neighbor_y) not in visited and abs(image_slice[x, y] - image_slice[neighbor_x, neighbor_y]) < threshold:
                                # Agregar el vecino a la región
                                region.append((neighbor_x, neighbor_y))

            # Crear una matriz de ceros con las mismas dimensiones que la imagen
            segmented_image = np.zeros_like(image_slice)
            # Asignar valor 255 a los píxeles de la región
            for x, y in visited:
                segmented_image[x, y] = 255

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

    # Configurar tamaño de la ventana y centrarla en la pantalla
    window_width = 800
    window_height = 600
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x = (screen_width / 2) - (window_width / 2)
    y = (screen_height / 2) - (window_height / 2)

    root.geometry("%dx%d+%d+%d" % (window_width, window_height, x, y))

    # Hacer la ventana principal un poco más grande
    root.update_idletasks()
    root.minsize(root.winfo_reqwidth() + 50, root.winfo_reqheight() + 50)

    root.mainloop()

if __name__ == "__main__":
    main()
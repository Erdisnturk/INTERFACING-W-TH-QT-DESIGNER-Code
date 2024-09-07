# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:15:45 2024

@author: User
"""

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QWidget, QInputDialog
from PyQt5.QtGui import QPixmap, QImage
from Lab_Interface import Ui_MainWindow  # `interface.py` dosyasındaki sınıfı import edin
import matplotlib.pyplot as plt
from skimage import io, color, filters, segmentation, morphology, data
from skimage.filters import threshold_multiotsu, roberts, sobel, scharr, prewitt
from skimage.segmentation import chan_vese, morphological_chan_vese, inverse_gaussian_gradient, checkerboard_level_set
from skimage.morphology import disk
from skimage import img_as_ubyte, img_as_float
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import tempfile
from skimage.color import rgb2hsv
from PIL import Image

class Command:
    def execute(self):
        raise NotImplementedError("Subclasses should implement this method")

    def undo(self):
        raise NotImplementedError("Subclasses should implement this method")

class GrayscaleCommand(Command):
    def __init__(self, receiver):
        self.receiver = receiver
        self.prev_state = None

    def execute(self):
        self.prev_state = self.receiver.get_image_state()
        self.receiver.convert_to_grayscale()

    def undo(self):
        self.receiver.set_image_state(self.prev_state)

class HsvCommand(Command):
    def __init__(self, receiver):
        self.receiver = receiver
        self.prev_state = None

    def execute(self):
        self.prev_state = self.receiver.get_image_state()
        self.receiver.convert_to_hsv()

    def undo(self):
        self.receiver.set_image_state(self.prev_state)
class UndoManager:
    def __init__(self):
        self.items = []
        self.position = -1

    def undo(self):
        if self.position >= 0:
            item = self.items[self.position]
            if item.undo:
                item.undo()
            self.position -= 1

    def redo(self):
        if self.position < len(self.items) - 1:
            item = self.items[self.position + 1]
            if item.redo:
                item.redo()
            self.position += 1

    def clearUndo(self):
        self.items = self.items[self.position + 1:]
        self.position = -1

    def clearRedo(self):
        self.items = self.items[:self.position + 1]

    def addItem(self, item):
        self.clearRedo()
        self.items.append(item)
        self.position += 1

    def removeItem(self, index):
        if 0 <= index < len(self.items):
            del self.items[index]
            if index <= self.position:
                self.position -= 1

    def item(self, index):
        if 0 <= index < len(self.items):
            return self.items[index]
        return None

    @property
    def length(self):
        return len(self.items)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value


class MultiOtsuCommand(Command):
    def __init__(self, receiver):
        self.receiver = receiver
        self.prev_state = None

    def execute(self):
        self.prev_state = self.receiver.get_image_state()
        self.receiver.apply_multi_otsu()

    def undo(self):
        self.receiver.set_image_state(self.prev_state)

class ChanVeseCommand(Command):
    def __init__(self, receiver):
        self.receiver = receiver
        self.prev_state = None

    def execute(self):
        self.prev_state = self.receiver.get_image_state()
        self.receiver.apply_chan_vese()

    def undo(self):
        self.receiver.set_image_state(self.prev_state)

class MorphologicalACWECommand(Command):
    def __init__(self, receiver):
        self.receiver = receiver
        self.prev_state = None

    def execute(self):
        self.prev_state = self.receiver.get_image_state()
        self.receiver.apply_morphological_acwe()

    def undo(self):
        self.receiver.set_image_state(self.prev_state)

class EdgeDetectionCommand(Command):
    def __init__(self, receiver, method):
        self.receiver = receiver
        self.method = method
        self.prev_state = None

    def execute(self):
        self.prev_state = self.receiver.get_image_state()
        self.receiver.apply_edge_detection(self.method)

    def undo(self):
        self.receiver.set_image_state(self.prev_state)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        self.commands = []
        self.current_command_index = -1
        self.source_image = None
        self.output_image = None

        # Başlangıçta sadece Open Source işlevi etkin olacak
        self.actionSave_Output.setEnabled(False)
        self.actionSave_As_Output.setEnabled(False)
        self.actionSource_Output.setEnabled(False)
        self.actionOutput.setEnabled(False)
        self.actionExit.setEnabled(True)

        self.Export_As_Source_Button.setEnabled(False)
        self.Save_Output_Button.setEnabled(False)
        self.Save_As_Button.setEnabled(False)
        self.Export_As_Output_Button.setEnabled(False)
        self.Output_Button.setEnabled(False)

        self.actionSource.setEnabled(False)
        self.actionOutput_2.setEnabled(False)
        self.actionUndo_Output.setEnabled(False)
        self.actionRedo_Output.setEnabled(False)

        # Menü ve butonları bağlama
        self.Open_Source_Button.clicked.connect(self.open_source)
        self.Open_Source.triggered.connect(self.open_source)

        self.Export_As_Source_Button.clicked.connect(self.export_as_source)
        self.actionSource_Output.triggered.connect(self.export_as_source)

        self.Source_Button.clicked.connect(self.clear_source)
        self.actionSource.triggered.connect(self.clear_source)

        self.Save_Output_Button.clicked.connect(self.save_output)
        self.actionSave_Output.triggered.connect(self.save_output)

        self.Save_As_Button.clicked.connect(self.save_as_output)
        self.actionSave_As_Output.triggered.connect(self.save_as_output)

        self.Export_As_Output_Button.clicked.connect(self.export_as_output)
        self.actionOutput.triggered.connect(self.export_as_output)

        self.Output_Button.clicked.connect(self.clear_output)
        self.actionOutput_2.triggered.connect(self.clear_output)

        self.Undo_Button.clicked.connect(self.undo_output)
        self.actionUndo_Output.triggered.connect(self.undo_output)

        self.Rendo_Button.clicked.connect(self.redo_output)
        self.actionRedo_Output.triggered.connect(self.redo_output)

        self.RGB_to_Grayscale_Button.clicked.connect(self.rgb_to_grayscale)
        self.actionRGB_to_Grayscale.triggered.connect(self.rgb_to_grayscale)

        self.RGB_to_HSV_Button.clicked.connect(self.rgb_to_hsv)
        self.actionRGB_to_HSV.triggered.connect(self.rgb_to_hsv)

        self.Multi_Atsu_Button.clicked.connect(self.multi_otsu_thresholding)
        self.actionMulti_Otsu_Thresholding.triggered.connect(self.multi_otsu_thresholding)

        self.Chan_Vese_Button.clicked.connect(self.chan_vese_segmentation)
        self.actionChan_Vese_Segmentation.triggered.connect(self.chan_vese_segmentation)

        self.Morphological_Button.clicked.connect(self.apply_morphological_acwe)
        self.actionMorphological_Snakes.triggered.connect(self.apply_morphological_acwe)

        self.pushButton_18.clicked.connect(self.apply_roberts)
        self.actionRoberts.triggered.connect(self.apply_roberts)

        self.pushButton_19.clicked.connect(self.apply_sobel)
        self.actionSobel.triggered.connect(self.apply_sobel)

        self.pushButton_20.clicked.connect(self.apply_scharr)
        self.actionScharr.triggered.connect(self.apply_scharr)

    def enable_post_open_features(self):
        self.actionSave_Output.setEnabled(True)
        self.actionSave_As_Output.setEnabled(True)
        self.actionSource_Output.setEnabled(True)
        self.actionOutput.setEnabled(True)
        self.Export_As_Source_Button.setEnabled(True)
        self.Save_Output_Button.setEnabled(True)
        self.Save_As_Button.setEnabled(True)
        self.Export_As_Output_Button.setEnabled(True)
        self.Output_Button.setEnabled(True)
        self.actionSource.setEnabled(True)

    def enable_post_operation_features(self):
        self.actionUndo_Output.setEnabled(True)
        self.actionRedo_Output.setEnabled(True)
        self.actionOutput_2.setEnabled(True)

    def open_source(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Source Image", "", "Image Files (*.png *.jpg);;All Files (*)", options=options)
        if fileName:
            self.source_image = Image.open(fileName)
            self.display_image(self.Source_Label, self.source_image)
            self.enable_post_open_features()

    def export_as_source(self):
        pass

    def clear_source(self):
        self.Source_Label.setPixmap(QPixmap())
        self.Source_Label.setText("")
        self.clear_output()

    def save_output(self):
        pass

    def save_as_output(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save As Output Image", "", "Image Files (*.png *.jpg);;All Files (*)", options=options)
        if fileName:
            io.imsave(fileName, img_as_ubyte(self.output_image))

    def export_as_output(self):
        pass

    def clear_output(self):
        self.Output_Label.setPixmap(QPixmap())
        self.Output_Label.setText("")

    def undo_output(self):
        if self.current_command_index >= 0:
            self.commands[self.current_command_index].undo()
            self.current_command_index -= 1

    def redo_output(self):
        if self.current_command_index < len(self.commands) - 1:
            self.current_command_index += 1
            self.commands[self.current_command_index].execute()

    def rgb_to_grayscale(self):
        command = GrayscaleCommand(self)
        command.execute()
        self.commands.append(command)
        self.current_command_index += 1
        self.enable_post_operation_features()

    def rgb_to_hsv(self):
        command = HsvCommand(self)
        command.execute()
        self.commands.append(command)
        self.current_command_index += 1
        self.enable_post_operation_features()

    def multi_otsu_thresholding(self):
        command = MultiOtsuCommand(self)
        command.execute()
        self.commands.append(command)
        self.current_command_index += 1
        self.enable_post_operation_features()

    def chan_vese_segmentation(self):
        command = ChanVeseCommand(self)
        command.execute()
        self.commands.append(command)
        self.current_command_index += 1
        self.enable_post_operation_features()

    def apply_morphological_acwe(self):
        command = MorphologicalACWECommand(self)
        command.execute()
        self.commands.append(command)
        self.current_command_index += 1
        self.enable_post_operation_features()

    def apply_roberts(self):
        command = EdgeDetectionCommand(self, method="roberts")
        command.execute()
        self.commands.append(command)
        self.current_command_index += 1
        self.enable_post_operation_features()

    def apply_sobel(self):
        command = EdgeDetectionCommand(self, method="sobel")
        command.execute()
        self.commands.append(command)
        self.current_command_index += 1
        self.enable_post_operation_features()

    def apply_scharr(self):
        command = EdgeDetectionCommand(self, method="scharr")
        command.execute()
        self.commands.append(command)
        self.current_command_index += 1
        self.enable_post_operation_features()

    def apply_prewitt(self):
        command = EdgeDetectionCommand(self, method="prewitt")
        command.execute()
        self.commands.append(command)
        self.current_command_index += 1
        self.enable_post_operation_features()

    def convert_to_grayscale(self):
        self.output_image = color.rgb2gray(np.array(self.source_image))
        self.display_image(self.Output_Label, Image.fromarray((self.output_image * 255).astype(np.uint8)), is_gray=True)

    def convert_to_hsv(self):
        if self.source_image is not None:
            # Convert the PIL image to a NumPy array
            source_array = np.array(self.source_image)
            
            # Perform the RGB to HSV conversion
            hsv_img = color.rgb2hsv(source_array)
            
            # Extract the hue and value channels
            hue_img = hsv_img[:, :, 0]
            value_img = hsv_img[:, :, 2]
            
            # Apply the thresholds
            hue_threshold = 0.04
            value_threshold = 0.10
            binary_img = (hue_img > hue_threshold) | (value_img < value_threshold)
            
            # Display the binary image using matplotlib
            fig, ax0 = plt.subplots(figsize=(10, 8))
            ax0.imshow(binary_img)
            ax0.axis('off')
            fig.tight_layout()
            plt.show()
            
            # Display the binary image in the Output_Label
            self.display_matplotlib_image(self.Output_Label, fig)
            
            # Enable relevant actions
            self.actionSave_Output.setEnabled(True)
            self.actionSave_As_Output.setEnabled(True)
            self.actionOutput.setEnabled(True)
            self.actionUndo_Output.setEnabled(True)
            self.actionRedo_Output.setEnabled(True)
            self.add_to_history(binary_img)

    def apply_multi_otsu(self):
        if self.source_image is not None:
            # Convert the PIL image to a NumPy array
            source_array = np.array(self.source_image)
            
            # Apply Multi-Otsu thresholding
            thresholds = threshold_multiotsu(source_array)
            regions = np.digitize(source_array, bins=thresholds)
            
            # Display the regions using matplotlib
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(regions, cmap='jet')
            ax.axis('off')
            plt.show()
            
            # Display the regions in the Output_Label
            self.display_matplotlib_image(self.Output_Label, fig)
            
            # Enable relevant actions
            self.actionSave_Output.setEnabled(True)
            self.actionSave_As_Output.setEnabled(True)
            self.actionOutput.setEnabled(True)
            self.actionUndo_Output.setEnabled(True)
            self.actionRedo_Output.setEnabled(True)
            self.add_to_history(regions)

    def apply_chan_vese(self):
        self.output_image = chan_vese(np.array(self.source_image), mu=0.25, lambda1=1, lambda2=1, tol=1e-8, max_num_iter=200, dt=0.5, init_level_set="checkerboard", extended_output=False)
        self.display_image(self.Output_Label, Image.fromarray((self.output_image * 255).astype(np.uint8)))

    def apply_morphological_acwe(self):
        image_acwe = img_as_float(np.array(self.source_image))
        init_ls_acwe = checkerboard_level_set(image_acwe.shape, 6)
        evolution_acwe = []
        callback_acwe = store_evolution_in(evolution_acwe)
        ls_acwe = morphological_chan_vese(
            image_acwe, num_iter=35, init_level_set=init_ls_acwe, smoothing=3, iter_callback=callback_acwe
        )

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(image_acwe, cmap="gray")
        ax.set_axis_off()
        ax.contour(ls_acwe, [0.5], colors='r')

        plt.show()
        self.display_matplotlib_image(self.Output_Label,fig)

    def apply_edge_detection(self, method):
        image = np.array(self.source_image)
        if method == "roberts":
            self.output_image = roberts(image)
        elif method == "sobel":
            self.output_image = sobel(image)
        elif method == "scharr":
            self.output_image = scharr(image)
        elif method == "prewitt":
            self.output_image = prewitt(image)
        self.display_image(self.Output_Label, Image.fromarray((self.output_image * 255).astype(np.uint8)), is_gray=True)

    def display_image(self, label, image, is_gray=False):
        if isinstance(image, Image.Image):
            image = image.convert("RGBA")
            qimage = QImage(image.tobytes(), image.width, image.height, QImage.Format_RGBA8888)
        else:
            if is_gray: 
                plt.imsave('temp.png', image, cmap='gray')
            else:
                plt.imsave('temp.png', image)
            qimage = QImage('temp.png')
        pixmap = QPixmap.fromImage(qimage)
        label.setPixmap(pixmap)
        label.setScaledContents(True)  # Resmi sığdırmak için ekledik
        self.adjustSize()  # Pencereyi resme göre yeniden boyutlandırmak için ekledik

    def display_matplotlib_image(self, label, fig):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            fig.savefig(temp_file.name)
            temp_file.close()
            pixmap = QPixmap(temp_file.name)
            label.setPixmap(pixmap)
            label.setScaledContents(True)
            self.adjustSize()

    def get_image_state(self):
        return self.output_image

    def set_image_state(self, state):
        self.output_image = state
        self.display_image(self.Output_Label, Image.fromarray((self.output_image * 255).astype(np.uint8)))

def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """
    def _store(x):
        lst.append(np.copy(x))
    return _store

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
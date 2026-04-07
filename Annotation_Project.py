import sys
import os
import cv2
import numpy as np
from pathlib import Path
import torch
import re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                             QHBoxLayout, QWidget, QLabel, QFileDialog, QMessageBox,
                             QScrollArea, QRadioButton, QButtonGroup, QSizePolicy, QAction, QSlider,
                             QInputDialog)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint, QTimer
import copy

_original_load = torch.load
def legacy_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = legacy_load

BASE_DIR = Path(__file__).resolve().parent
TEMP_FILE = BASE_DIR / "last_session.txt"
COLORS = [(0,255,0),(0,0,255),(255,0,0),(0,165,255),(255,255,0),(255,0,255),(128,0,128),(0,128,128)]
MAX_UNDO = 30


class AnnotationRow(QWidget):
    def __init__(self, index, cls, is_selected, on_delete, on_change, on_select, class_names):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        if is_selected:
            self.setStyleSheet("background-color: #008080; border-radius: 4px; color: white;")

        lbl_layout = QHBoxLayout()
        self.label = QLabel(f"<b>[{index+1}] Box</b>")
        self.label.mousePressEvent = lambda e: on_select(index)
        self.del_btn = QPushButton("Delete")
        self.del_btn.setFocusPolicy(Qt.NoFocus)
        self.del_btn.setStyleSheet("background-color: #c62828; color: white;")
        self.del_btn.clicked.connect(lambda: on_delete(index))
        lbl_layout.addWidget(self.label)
        lbl_layout.addWidget(self.del_btn)
        layout.addLayout(lbl_layout)

        btn_layout = QHBoxLayout()
        for i, name in class_names.items():
            btn = QPushButton(name)
            btn.setFocusPolicy(Qt.NoFocus)
            if cls == i:
                btn.setStyleSheet("background-color: #4CAF50; color: white;")
            btn.clicked.connect(lambda checked, idx=i: on_change(index, idx))
            btn_layout.addWidget(btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)


class FolderAnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dynamic YOLO Annotation Tool")
        self.showMaximized()

        self.model = None
        self.model_ext = None
        self.class_names = {}
        self.class_colors = {}

        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        self.image_paths = []
        self.current_idx = -1
        self.current_frame_clean = None
        self.boxes = []
        self.selected_idx = -1
        self._undo_stack = []

        self.draw_mode = False
        self.drawing = False
        self.resizing = False
        self.dragging = False
        self.panning = False
        self.hide_boxes = False
        self.use_model = True
        self.show_conf = False
        self.resize_edge = None
        self.last_pan_pos = QPoint()

        self.start_point = QPoint()
        self.end_point = QPoint()
        self.zoom_factor = 1.0
        self.zoom_center = QPoint(0, 0)
        self.FOCUS_ZOOM_PADDING = 2.5

        self.create_menu_bar()
        self.initUI()
        self.showMaximized()

    def load_last_filename(self):
        if TEMP_FILE.exists():
            with open(TEMP_FILE, "r") as f:
                return f.read().strip()
        return ""

    def save_last_filename(self, filename):
        with open(TEMP_FILE, "w") as f:
            f.write(filename)

    def _push_undo(self):
        self._undo_stack.append(copy.deepcopy(self.boxes))
        if len(self._undo_stack) > MAX_UNDO:
            self._undo_stack.pop(0)

    def undo(self):
        if not self._undo_stack:
            return
        self.boxes = self._undo_stack.pop()
        self.selected_idx = -1
        self.refresh_side_panel()
        self.update_display()

    def init_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model", "",
                                              "Models (*.pt *.tflite *.onnx *.pb)")
        if not path:
            return
        self.model_ext = Path(path).suffix.lower()
        try:
            if self.model_ext == '.pt':
                from ultralytics import YOLO
                self.model = YOLO(path)
                self.class_names = self.model.names
            elif self.model_ext == '.tflite':
                import tensorflow as tf
                self.model = tf.lite.Interpreter(model_path=path)
                self.model.allocate_tensors()
            elif self.model_ext in ['.onnx', '.pb']:
                self.model = cv2.dnn.readNet(path)
                self._onnx_input_size = (640, 640)

            self.class_colors = {i: COLORS[i % len(COLORS)] for i in self.class_names}
            if hasattr(self, 'class_selector_layout'):
                self.rebuild_class_selector()
                self.refresh_side_panel()
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {e}")

    def load_classes(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Classes File", "",
                                              "Text Files (*.txt)")
        if not path:
            return
        try:
            with open(path, "r") as f:
                names = [l.strip() for l in f if l.strip()]
            self.class_names = {i: name for i, name in enumerate(names)}
            self.class_colors = {i: COLORS[i % len(COLORS)] for i in self.class_names}
            if hasattr(self, 'class_selector_layout'):
                self.rebuild_class_selector()
                self.refresh_side_panel()
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load classes: {e}")

    def create_menu_bar(self):
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        file_menu = menubar.addMenu("File")

        act = QAction("Load Folder", self)
        act.setShortcut("Ctrl+O")
        act.triggered.connect(self.load_folder)
        file_menu.addAction(act)

        act = QAction("Load Object Detection Model  (M)", self)
        act.setShortcut("M")
        act.triggered.connect(self.init_model)
        file_menu.addAction(act)

        act = QAction("Load Classes (.txt)  (C)", self)
        act.setShortcut("C")
        act.triggered.connect(self.load_classes)
        file_menu.addAction(act)

        act = QAction("Exit", self)
        act.setShortcut("Ctrl+Q")
        act.triggered.connect(self.close)
        file_menu.addAction(act)

        view_menu = menubar.addMenu("View")
        self.toggle_model_action = QAction("Enable Model Inference", self, checkable=True)
        self.toggle_model_action.setChecked(self.use_model)
        self.toggle_model_action.triggered.connect(self.toggle_model_inference)
        view_menu.addAction(self.toggle_model_action)

        self.toggle_hide_action = QAction("Hide Bounding Boxes", self, checkable=True)
        self.toggle_hide_action.setShortcut("H")
        self.toggle_hide_action.setChecked(self.hide_boxes)
        self.toggle_hide_action.triggered.connect(self.toggle_hide)
        view_menu.addAction(self.toggle_hide_action)

        edit_menu = menubar.addMenu("Edit")
        undo_act = QAction("Undo", self)
        undo_act.setShortcut("Ctrl+Z")
        undo_act.triggered.connect(self.undo)
        edit_menu.addAction(undo_act)

        nav_menu = menubar.addMenu("Navigation")
        goto_act = QAction("Go to Frame  (G)", self)
        goto_act.setShortcut("G")
        goto_act.triggered.connect(self.goto_frame)
        nav_menu.addAction(goto_act)

    def initUI(self):
        self.central_widget = QWidget()
        self.central_widget.setMouseTracking(True)
        self.setCentralWidget(self.central_widget)
        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        self.image_label = QLabel("Select File -> Load Folder to begin")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #111; border: 1px solid #444; color: white;")
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setMouseTracking(True)
        left_layout.addWidget(self.image_label, stretch=1)

        bottom_btns = QHBoxLayout()
        self.mode_btn = QPushButton("Mode: Move/Edit (Space)")
        self.mode_btn.setCheckable(True)
        self.mode_btn.setFocusPolicy(Qt.NoFocus)
        self.mode_btn.clicked.connect(self.toggle_mode)

        self.prev_btn = QPushButton("Prev (A)")
        self.prev_btn.setFocusPolicy(Qt.NoFocus)
        self.prev_btn.clicked.connect(self.prev_image)

        self.skip_btn = QPushButton("Skip (D)")
        self.skip_btn.setFocusPolicy(Qt.NoFocus)
        self.skip_btn.clicked.connect(self.next_image)

        self.del_frame_btn = QPushButton("Del Frame (X)")
        self.del_frame_btn.setFocusPolicy(Qt.NoFocus)
        self.del_frame_btn.setStyleSheet("background-color: #c62828; color: white;")
        self.del_frame_btn.clicked.connect(self.delete_frame)

        self.save_btn = QPushButton("Save & Next (S)")
        self.save_btn.setFocusPolicy(Qt.NoFocus)
        self.save_btn.setStyleSheet("background-color: #2e7d32; color: white;")
        self.save_btn.clicked.connect(self.save_and_next)

        for w in [self.mode_btn, self.prev_btn, self.skip_btn, self.del_frame_btn, self.save_btn]:
            bottom_btns.addWidget(w)
        left_layout.addLayout(bottom_btns)

        self.side_panel = QWidget()
        self.side_panel.setFixedWidth(380)
        side_layout = QVBoxLayout()

        self.status_info = QLabel("<b>Progress: 0 / 0</b>")
        self.status_info.setStyleSheet("font-size: 14px; padding: 5px;")
        side_layout.addWidget(self.status_info)

        side_layout.addWidget(QLabel("Confidence Threshold (Model):"))
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(0, 100)
        self.thresh_slider.setValue(50)
        self.thresh_slider.setFocusPolicy(Qt.NoFocus)
        side_layout.addWidget(self.thresh_slider)

        self.class_selector_layout = QHBoxLayout()
        self.class_group = QButtonGroup()
        self.rebuild_class_selector()
        side_layout.addLayout(self.class_selector_layout)

        self.scroll = QScrollArea()
        self.scroll.setFocusPolicy(Qt.NoFocus)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.scroll_widget)
        side_layout.addWidget(self.scroll)

        self.side_panel.setLayout(side_layout)
        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addWidget(self.side_panel)
        self.central_widget.setLayout(main_layout)

    def rebuild_class_selector(self):
        while self.class_selector_layout.count():
            child = self.class_selector_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.class_group = QButtonGroup()
        for i, name in self.class_names.items():
            rb = QRadioButton(name)
            rb.setFocusPolicy(Qt.NoFocus)
            if i == 0:
                rb.setChecked(True)
            self.class_group.addButton(rb, i)
            self.class_selector_layout.addWidget(rb)

    def _sync_class_selector(self, cls_id):
        btn = self.class_group.button(cls_id)
        if btn:
            btn.setChecked(True)
        else:
            buttons = self.class_group.buttons()
            if buttons:
                buttons[0].setChecked(True)

    def toggle_mode(self):
        self.draw_mode = self.mode_btn.isChecked()
        if self.draw_mode:
            self.mode_btn.setText("Mode: DRAWING (Space)")
            self.mode_btn.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold;")
            self.setCursor(Qt.CrossCursor)
        else:
            self.mode_btn.setText("Mode: Move/Edit (Space)")
            self.mode_btn.setStyleSheet("")
            self.setCursor(Qt.ArrowCursor)
        self.setFocus()

    def toggle_model_inference(self, checked):
        self.use_model = checked

    def toggle_hide(self, checked):
        self.hide_boxes = checked
        self.update_display()

    @staticmethod
    def _natural_key(path):
        name = os.path.basename(path)
        parts = re.split(r'(\d+)', name)
        return [int(p) if p.isdigit() else p.lower() for p in parts]

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        valid = {'.jpg', '.jpeg', '.png', '.bmp'}
        files = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in valid
        ]
        self.image_paths = sorted(files, key=self._natural_key)
        if self.image_paths:
            last_file = self.load_last_filename()
            self.current_idx = 0
            for i, p in enumerate(self.image_paths):
                if os.path.basename(p) == last_file:
                    self.current_idx = i
                    break
            self.load_image()

    def load_image(self):
        if not (0 <= self.current_idx < len(self.image_paths)):
            return
        img_path = self.image_paths[self.current_idx]
        frame = cv2.imread(img_path)
        if frame is None:
            QMessageBox.warning(self, "Warning", f"Cannot read image:\n{img_path}")
            return
        self.current_frame_clean = frame
        h_img, w_img = frame.shape[:2]
        self._undo_stack.clear()
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        self.boxes = []
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, x_c, y_c, w, h = map(float, parts[:5])
                        x1 = int((x_c - w / 2) * w_img)
                        y1 = int((y_c - h / 2) * h_img)
                        x2 = int((x_c + w / 2) * w_img)
                        y2 = int((y_c + h / 2) * h_img)
                        self.boxes.append([x1, y1, x2, y2, int(cls)])
        elif self.use_model and self.model is not None:
            self._run_inference(frame, w_img, h_img)
        self.selected_idx = -1
        self.zoom_factor = 1.0
        self.zoom_center = QPoint(w_img // 2, h_img // 2)
        self.status_info.setText(
            f"<b>Progress: {self.current_idx + 1} / {len(self.image_paths)}</b>"
            f"<br>{os.path.basename(img_path)}"
        )
        self.refresh_side_panel()
        self.update_display()

    def _run_inference(self, frame, w_img, h_img):
        try:
            conf_thresh = self.thresh_slider.value() / 100.0
            if self.model_ext == '.pt':
                results = self.model(frame, conf=conf_thresh, verbose=False)[0]
                for b in results.boxes:
                    c = b.xyxy[0].cpu().numpy()
                    self.boxes.append([int(c[0]), int(c[1]), int(c[2]), int(c[3]),
                                       int(b.cls[0]), float(b.conf[0])])
            elif self.model_ext == '.tflite':
                inp = self.model.get_input_details()
                out = self.model.get_output_details()
                th, tw = inp[0]['shape'][1:3]
                img_r = cv2.resize(frame, (tw, th))
                data = np.expand_dims(img_r, 0).astype(inp[0]['dtype'])
                if inp[0]['dtype'] == np.float32:
                    data = (data.astype(np.float32) - 127.5) / 127.5
                self.model.set_tensor(inp[0]['index'], data)
                self.model.invoke()
                boxes   = self.model.get_tensor(out[0]['index'])[0]
                classes = self.model.get_tensor(out[1]['index'])[0]
                scores  = self.model.get_tensor(out[2]['index'])[0]
                for i in range(len(scores)):
                    if scores[i] > conf_thresh:
                        ymin, xmin, ymax, xmax = boxes[i]
                        self.boxes.append([int(xmin*w_img), int(ymin*h_img),
                                           int(xmax*w_img), int(ymax*h_img),
                                           int(classes[i]), float(scores[i])])
            elif self.model_ext in ['.onnx', '.pb']:
                iw, ih = getattr(self, '_onnx_input_size', (640, 640))
                blob = cv2.dnn.blobFromImage(frame, 1/255.0, (iw, ih), swapRB=True, crop=False)
                self.model.setInput(blob)
                outs = self.model.forward()
                if len(outs.shape) == 3:
                    if outs.shape[1] < outs.shape[2]:
                        outs = outs.transpose(0, 2, 1)
                    for row in outs[0]:
                        scores = row[4:]
                        if not len(scores): continue
                        cid = int(np.argmax(scores))
                        conf = float(scores[cid])
                        if conf > conf_thresh:
                            cx, cy, bw, bh = row[:4]
                            x1 = int((cx - bw/2) * w_img / iw)
                            y1 = int((cy - bh/2) * h_img / ih)
                            x2 = int((cx + bw/2) * w_img / iw)
                            y2 = int((cy + bh/2) * h_img / ih)
                            self.boxes.append([x1, y1, x2, y2, cid, conf])
        except Exception as e:
            QMessageBox.warning(self, "Inference error", str(e))

    def prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image()

    def next_image(self):
        if self.current_idx < len(self.image_paths) - 1:
            self.current_idx += 1
            self.load_image()

    def goto_frame(self):
        if not self.image_paths:
            QMessageBox.warning(self, "Warning", "No images loaded. Load a folder first.")
            return
        total_frames = len(self.image_paths)
        frame_num, ok = QInputDialog.getInt(
            self, "Go to Frame",
            f"Enter frame number (1-{total_frames}):",
            self.current_idx + 1,
            1,
            total_frames,
            1
        )
        if ok:
            self.current_idx = frame_num - 1
            self.load_image()

    def delete_frame(self):
        if not (0 <= self.current_idx < len(self.image_paths)):
            return
        img_path = self.image_paths[self.current_idx]
        reply = QMessageBox.question(
            self, "Delete frame",
            f"Permanently delete\n{os.path.basename(img_path)}\nand its label file?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        for p in (img_path, txt_path):
            if os.path.exists(p):
                os.remove(p)
        self.image_paths.pop(self.current_idx)
        if not self.image_paths:
            self.current_frame_clean = None
            self.image_label.clear()
            self.image_label.setText("Select File -> Load Folder to begin")
        else:
            self.current_idx = min(self.current_idx, len(self.image_paths) - 1)
            self.load_image()

    def keyPressEvent(self, event):
        k = event.key()
        if k in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            event.accept()
            self._handle_arrow_key(k)
            return
        if k == Qt.Key_S:
            self.save_and_next()
        elif k == Qt.Key_D:
            self.next_image()
        elif k == Qt.Key_A:
            self.prev_image()
        elif k == Qt.Key_X:
            self.delete_frame()
        elif k == Qt.Key_H:
            self.toggle_hide_action.setChecked(not self.toggle_hide_action.isChecked())
            self.toggle_hide(self.toggle_hide_action.isChecked())
        elif k == Qt.Key_Space:
            self.mode_btn.setChecked(not self.mode_btn.isChecked())
            self.toggle_mode()
        elif k in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.selected_idx != -1:
                self.delete_box(self.selected_idx)
        elif k == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            self.undo()
        elif k == Qt.Key_M:
            self.init_model()
        elif k == Qt.Key_C:
            self.load_classes()
        elif k == Qt.Key_G:
            self.goto_frame()
        elif k == Qt.Key_BracketLeft:
            self.thresh_slider.setValue(max(0, self.thresh_slider.value() - 5))
        elif k == Qt.Key_BracketRight:
            self.thresh_slider.setValue(min(100, self.thresh_slider.value() + 5))
        elif Qt.Key_1 <= k <= Qt.Key_9:
            cls_idx = k - Qt.Key_1
            if cls_idx in self.class_names:
                self._sync_class_selector(cls_idx)
                if self.selected_idx != -1:
                    self.change_class(self.selected_idx, cls_idx)
        event.accept()

    def _get_transform(self):
        w_lbl = max(1, self.image_label.width())
        h_lbl = max(1, self.image_label.height())
        h_img, w_img = self.current_frame_clean.shape[:2]
        
        base_scale = min(w_lbl / w_img, h_lbl / h_img)
        scale = base_scale * self.zoom_factor
        
        scaled_w = int(w_img * scale)
        scaled_h = int(h_img * scale)
        
        zx_scaled = int(self.zoom_center.x() * scale)
        zy_scaled = int(self.zoom_center.y() * scale)
        
        ox = w_lbl // 2 - zx_scaled
        oy = h_lbl // 2 - zy_scaled
        
        if scaled_w > w_lbl:
            ox = min(0, max(w_lbl - scaled_w, ox))
        else:
            ox = (w_lbl - scaled_w) // 2
            
        if scaled_h > h_lbl:
            oy = min(0, max(h_lbl - scaled_h, oy))
        else:
            oy = (h_lbl - scaled_h) // 2
            
        return scale, ox, oy

    def _handle_arrow_key(self, k):
        if self.current_frame_clean is None:
            return
        if self.selected_idx != -1:
            step = 5
            self._push_undo()
            b = self.boxes[self.selected_idx]
            dx = dy = 0
            if k == Qt.Key_Left:    dx = -step
            elif k == Qt.Key_Right: dx =  step
            elif k == Qt.Key_Up:    dy = -step
            elif k == Qt.Key_Down:  dy =  step
            h_img, w_img = self.current_frame_clean.shape[:2]
            bw, bh = b[2]-b[0], b[3]-b[1]
            b[0] = max(0, min(b[0]+dx, w_img-bw))
            b[2] = b[0] + bw
            b[1] = max(0, min(b[1]+dy, h_img-bh))
            b[3] = b[1] + bh
            self.update_display()
        elif self.zoom_factor > 1.0:
            scale, _, _ = self._get_transform()
            pan = max(5, int(40 / scale))
            if k == Qt.Key_Left:    self.zoom_center.setX(self.zoom_center.x() - pan)
            elif k == Qt.Key_Right: self.zoom_center.setX(self.zoom_center.x() + pan)
            elif k == Qt.Key_Up:    self.zoom_center.setY(self.zoom_center.y() - pan)
            elif k == Qt.Key_Down:  self.zoom_center.setY(self.zoom_center.y() + pan)
            self._clamp_zoom_center()
            self.update_display()

    def _clamp_zoom_center(self):
        if self.current_frame_clean is None:
            return
        h_img, w_img = self.current_frame_clean.shape[:2]
        cx = max(0, min(self.zoom_center.x(), w_img))
        cy = max(0, min(self.zoom_center.y(), h_img))
        self.zoom_center = QPoint(cx, cy)

    def map_to_image(self, pos):
        lbl_pos = self.image_label.mapFrom(self, pos)
        scale, ox, oy = self._get_transform()
        h_img, w_img = self.current_frame_clean.shape[:2]
        
        x = int((lbl_pos.x() - ox) / scale)
        y = int((lbl_pos.y() - oy) / scale)
        
        return QPoint(max(0, min(x, w_img)), max(0, min(y, h_img)))

    def get_edge_at(self, pos, box):
        x1, y1, x2, y2 = box[:4]
        margin = max(8, int(15 / self.zoom_factor))
        in_x = (x1 - margin) <= pos.x() <= (x2 + margin)
        in_y = (y1 - margin) <= pos.y() <= (y2 + margin)
        if in_y and abs(pos.x() - x1) <= margin: return 'left'
        if in_y and abs(pos.x() - x2) <= margin: return 'right'
        if in_x and abs(pos.y() - y1) <= margin: return 'top'
        if in_x and abs(pos.y() - y2) <= margin: return 'bottom'
        return None

    def _focus_on_box(self, idx):
        if self.current_frame_clean is None or not (0 <= idx < len(self.boxes)):
            return

        b = self.boxes[idx]
        h_img, w_img = self.current_frame_clean.shape[:2]

        box_cx = (b[0] + b[2]) / 2.0
        box_cy = (b[1] + b[3]) / 2.0

        box_w = max(1, b[2] - b[0])
        box_h = max(1, b[3] - b[1])

        w_lbl = max(1, self.image_label.width())
        h_lbl = max(1, self.image_label.height())

        pad = self.FOCUS_ZOOM_PADDING
        base_scale = min(w_lbl / w_img, h_lbl / h_img)
        target_scale = min(w_lbl / (box_w * pad), h_lbl / (box_h * pad))
        target_zoom = target_scale / base_scale

        self.zoom_factor = max(1.0, min(12.0, target_zoom))
        self.zoom_center = QPoint(int(box_cx), int(box_cy))
        self._clamp_zoom_center()

    def mousePressEvent(self, event):
        self.setFocus()
        if self.current_frame_clean is None:
            return

        if event.button() == Qt.RightButton:
            self.panning = True
            self.last_pan_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() != Qt.LeftButton:
            return

        img_pos = self.map_to_image(event.pos())

        if self.selected_idx != -1 and not self.draw_mode:
            edge = self.get_edge_at(img_pos, self.boxes[self.selected_idx])
            if edge:
                self._push_undo()
                self.resizing = True
                self.resize_edge = edge
                return

        clicked = False
        for i in range(len(self.boxes) - 1, -1, -1):
            if self.hide_boxes and i != self.selected_idx:
                continue
            b = self.boxes[i]
            if b[0] <= img_pos.x() <= b[2] and b[1] <= img_pos.y() <= b[3]:
                self.selected_idx = i
                self._sync_class_selector(b[4])
                if not self.draw_mode:
                    self._push_undo()
                    self.dragging = True
                    self.start_point = img_pos
                clicked = True
                break

        if not clicked:
            self.selected_idx = -1

        if not clicked and self.draw_mode:
            self.drawing = True
            self.start_point = img_pos
        elif not clicked and not self.draw_mode and self.zoom_factor > 1.0:
            self.panning = True
            self.last_pan_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        self.refresh_side_panel()
        if self.selected_idx != -1:
            idx_to_scroll = self.selected_idx
            QTimer.singleShot(50, lambda idx=idx_to_scroll: self._scroll_side_panel_to(idx))
        self.update_display()

    def mouseMoveEvent(self, event):
        if self.current_frame_clean is None:
            return
        if self.panning:
            scale, _, _ = self._get_transform()
            delta = event.pos() - self.last_pan_pos
            self.zoom_center.setX(int(self.zoom_center.x() - delta.x() / scale))
            self.zoom_center.setY(int(self.zoom_center.y() - delta.y() / scale))
            self._clamp_zoom_center()
            self.last_pan_pos = event.pos()
            self.update_display()
            return
            
        img_pos = self.map_to_image(event.pos())
        
        if self.resizing and self.selected_idx != -1:
            b = self.boxes[self.selected_idx]
            h_img, w_img = self.current_frame_clean.shape[:2]
            if self.resize_edge == 'left':   b[0] = max(0, min(img_pos.x(), b[2] - 5))
            elif self.resize_edge == 'right': b[2] = min(w_img, max(img_pos.x(), b[0] + 5))
            elif self.resize_edge == 'top':   b[1] = max(0, min(img_pos.y(), b[3] - 5))
            elif self.resize_edge == 'bottom':b[3] = min(h_img, max(img_pos.y(), b[1] + 5))
            self.update_display()
        elif self.dragging and self.selected_idx != -1 and not self.draw_mode:
            delta = img_pos - self.start_point
            b = self.boxes[self.selected_idx]
            h_img, w_img = self.current_frame_clean.shape[:2]
            bw, bh = b[2]-b[0], b[3]-b[1]
            new_x1 = max(0, min(b[0] + delta.x(), w_img - bw))
            new_y1 = max(0, min(b[1] + delta.y(), h_img - bh))
            b[0], b[1], b[2], b[3] = new_x1, new_y1, new_x1 + bw, new_y1 + bh
            self.start_point = img_pos
            self.update_display()
        elif self.drawing:
            self.end_point = img_pos
            self.update_display()
        else:
            if not self.draw_mode and self.selected_idx != -1:
                edge = self.get_edge_at(img_pos, self.boxes[self.selected_idx])
                if edge in ('left', 'right'):   self.setCursor(Qt.SizeHorCursor)
                elif edge in ('top', 'bottom'): self.setCursor(Qt.SizeVerCursor)
                else:                           self.setCursor(Qt.ArrowCursor)
            elif self.draw_mode:
                self.setCursor(Qt.CrossCursor)
            elif not self.draw_mode and self.zoom_factor > 1.0:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.RightButton, Qt.LeftButton) and self.panning:
            self.panning = False
            if self.draw_mode:
                self.setCursor(Qt.CrossCursor)
            elif self.zoom_factor > 1.0:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            if event.button() == Qt.RightButton:
                return
        if self.drawing:
            self.drawing = False
            self.end_point = self.map_to_image(event.pos())
            x1, y1 = self.start_point.x(), self.start_point.y()
            x2, y2 = self.end_point.x(), self.end_point.y()
            if abs(x1-x2) > 10 and abs(y1-y2) > 10:
                cid = self.class_group.checkedId()
                if cid == -1 and self.class_names:
                    cid = 0
                if cid != -1:
                    self._push_undo()
                    self.boxes.append([min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2), cid])
                    self.selected_idx = len(self.boxes) - 1
                    self.refresh_side_panel()
                    idx_to_scroll = self.selected_idx
                    QTimer.singleShot(50, lambda idx=idx_to_scroll: self._scroll_side_panel_to(idx))
        self.dragging = self.resizing = False
        self.resize_edge = None
        self.update_display()

    def wheelEvent(self, event):
        if self.current_frame_clean is None:
            return
        mouse_pos = self.map_to_image(event.pos())
        if event.angleDelta().y() > 0:
            self.zoom_factor = min(15.0, self.zoom_factor + 0.5)
            self.zoom_center = mouse_pos
        else:
            self.zoom_factor = max(1.0, self.zoom_factor - 0.5)
            if self.zoom_factor == 1.0:
                h_img, w_img = self.current_frame_clean.shape[:2]
                self.zoom_center = QPoint(w_img // 2, h_img // 2)
        self._clamp_zoom_center()
        self.update_display()

    def update_display(self):
        if self.current_frame_clean is None:
            return
        w_lbl = self.image_label.width()
        h_lbl = self.image_label.height()
        if w_lbl <= 0 or h_lbl <= 0:
            return

        scale, ox, oy = self._get_transform()
        disp = np.zeros((h_lbl, w_lbl, 3), dtype=np.uint8)
        
        h_img, w_img = self.current_frame_clean.shape[:2]
        scaled_w = int(w_img * scale)
        scaled_h = int(h_img * scale)
        
        dx1 = max(0, ox)
        dy1 = max(0, oy)
        dx2 = min(w_lbl, ox + scaled_w)
        dy2 = min(h_lbl, oy + scaled_h)
        
        sx1 = max(0, -ox)
        sy1 = max(0, -oy)
        sx2 = sx1 + (dx2 - dx1)
        sy2 = sy1 + (dy2 - dy1)
        
        if dx1 < dx2 and dy1 < dy2:
            ix1 = int(sx1 / scale)
            iy1 = int(sy1 / scale)
            ix2 = int(sx2 / scale)
            iy2 = int(sy2 / scale)
            
            crop = self.current_frame_clean[iy1:iy2, ix1:ix2]
            if crop.size > 0:
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                rw = dx2 - dx1
                rh = dy2 - dy1
                if rw > 0 and rh > 0:
                    disp[dy1:dy2, dx1:dx2] = cv2.resize(crop, (rw, rh), interpolation=interp)

        def ix(img_x): return int(img_x * scale + ox)
        def iy(img_y): return int(img_y * scale + oy)

        base_thick = max(1, int(self.zoom_factor * 0.7))
        font_scale = max(0.4, 0.5 * self.zoom_factor ** 0.4)

        for i, b in enumerate(self.boxes):
            if self.hide_boxes and i != self.selected_idx:
                continue
            color = (0, 0, 0) if i == self.selected_idx else self.class_colors.get(b[4], (0, 255, 0))
            thick = base_thick + 1 if i == self.selected_idx else base_thick
            bx1, by1, bx2, by2 = ix(b[0]), iy(b[1]), ix(b[2]), iy(b[3])
            cv2.rectangle(disp, (bx1, by1), (bx2, by2), color, thick)
            
            label_text = f"[{i+1}] {self.class_names.get(b[4], '')}"
            if self.show_conf and len(b) > 5:
                label_text += f" {b[5]:.2f}"
            if label_text:
                cv2.putText(disp, label_text, (bx1, max(0, by1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, max(1, thick))

        if self.drawing:
            cv2.rectangle(disp,
                          (ix(self.start_point.x()), iy(self.start_point.y())),
                          (ix(self.end_point.x()),   iy(self.end_point.y())),
                          (0, 0, 0), base_thick)

        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self.image_label.setPixmap(
            QPixmap.fromImage(
                QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                       rgb.shape[1] * 3, QImage.Format_RGB888)
            )
        )

    def refresh_side_panel(self):
        while self.scroll_layout.count():
            child = self.scroll_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        for i, b in enumerate(self.boxes):
            row = AnnotationRow(i, b[4], i == self.selected_idx,
                                self.delete_box, self.change_class,
                                self.select_box, self.class_names)
            self.scroll_layout.addWidget(row)

    def _scroll_side_panel_to(self, idx):
        QApplication.processEvents()
        item = self.scroll_layout.itemAt(idx)
        if item and item.widget():
            self.scroll.ensureWidgetVisible(item.widget())

    def select_box(self, idx):
        self.selected_idx = idx
        if 0 <= idx < len(self.boxes):
            self._sync_class_selector(self.boxes[idx][4])
            self._focus_on_box(idx)
        self.refresh_side_panel()
        QTimer.singleShot(50, lambda idx=idx: self._scroll_side_panel_to(idx))
        self.update_display()
        self.setFocus()

    def delete_box(self, idx):
        self._push_undo()
        self.boxes.pop(idx)
        self.selected_idx = -1
        self.refresh_side_panel()
        self.update_display()

    def change_class(self, box_idx, cls_idx):
        self._push_undo()
        self.boxes[box_idx][4] = cls_idx
        self._sync_class_selector(cls_idx)
        self.refresh_side_panel()
        self.update_display()

    def save_and_next(self):
        if self.current_frame_clean is None:
            return
        if not self.boxes:
            reply = QMessageBox.question(
                self, "No annotations",
                "No bounding boxes found. Save an empty label file anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        img_path = self.image_paths[self.current_idx]
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        h, w = self.current_frame_clean.shape[:2]
        with open(txt_path, "w") as f:
            for b in self.boxes:
                bw = b[2] - b[0]
                bh = b[3] - b[1]
                xc = (b[0] + bw / 2) / w
                yc = (b[1] + bh / 2) / h
                f.write(f"{b[4]} {xc:.6f} {yc:.6f} {bw/w:.6f} {bh/h:.6f}\n")
        self.save_last_filename(os.path.basename(img_path))
        self._undo_stack.clear()
        if self.current_idx < len(self.image_paths) - 1:
            self.current_idx += 1
            self.load_image()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FolderAnnotationTool()
    win.show()
    sys.exit(app.exec_())
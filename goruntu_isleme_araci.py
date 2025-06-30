import tkinter as tk
from tkinter import filedialog, simpledialog, Menu, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.morphology import skeletonize
import os

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Görüntü İşleme Arayüzü")
        self.root.geometry("1400x800")

        self.original_image = None
        self.processed_image = None
        self.is_binary = False

        self.create_widgets()

    def create_widgets(self):
        # Arayüz elemanlarının oluşturulduğu kısım (değişiklik yok)
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        image_frame = tk.Frame(main_frame)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        histogram_frame = tk.Frame(main_frame, width=400)
        histogram_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        histogram_frame.pack_propagate(False)
        original_frame = tk.LabelFrame(image_frame, text="Orijinal Görüntü (IO)", padx=5, pady=5)
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.original_label = tk.Label(original_frame)
        self.original_label.pack(fill=tk.BOTH, expand=True)
        self.original_label.bind("<Button-3>", lambda event: self.show_context_menu(event, 'original'))
        processed_frame = tk.LabelFrame(image_frame, text="İşlenmiş Görüntü", padx=5, pady=5)
        processed_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.processed_label = tk.Label(processed_frame)
        self.processed_label.pack(fill=tk.BOTH, expand=True)
        self.processed_label.bind("<Button-3>", lambda event: self.show_context_menu(event, 'processed'))
        histogram_label_frame = tk.LabelFrame(histogram_frame, text="Histogram Grafiği", padx=5, pady=5)
        histogram_label_frame.pack(fill=tk.BOTH, expand=True)
        self.fig = plt.Figure(figsize=(4, 3), dpi=100)
        self.hist_ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=histogram_label_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()
        self.create_menu()

    def create_menu(self):
        # Menülerin oluşturulduğu kısım (değişiklik yok)
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Dosya", menu=file_menu)
        file_menu.add_command(label="Görüntü Aç", command=self.load_image)
        file_menu.add_command(label="Sıfırla", command=self.reset_image)
        file_menu.add_separator()
        file_menu.add_command(label="Çıkış", command=self.root.quit)
        filter_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Filtreler", menu=filter_menu)
        filter_menu.add_command(label="Kenar Bulma (Sobel)", command=self.apply_edge_detection)
        filter_menu.add_command(label="Ortalama Filtresi", command=lambda: self.apply_filter_with_kernel('average'))
        filter_menu.add_command(label="Ortanca Filtresi", command=lambda: self.apply_filter_with_kernel('median'))
        filter_menu.add_command(label="Keskinleştirme", command=self.apply_sharpen)
        filter_menu.add_command(label="Yumuşatma (Gaussian Blur)", command=lambda: self.apply_filter_with_kernel('gaussian'))
        geo_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Geometrik İşlemler", menu=geo_menu)
        geo_menu.add_command(label="Döndür", command=self.apply_rotation)
        geo_menu.add_command(label="Aynalama (Yatay)", command=lambda: self.apply_mirror(1))
        geo_menu.add_command(label="Aynalama (Dikey)", command=lambda: self.apply_mirror(0))
        geo_menu.add_command(label="Shearing (Kaydırma)", command=self.apply_shearing)
        hist_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Histogram", menu=hist_menu)
        hist_menu.add_command(label="Histogram Eşitleme", command=self.apply_histogram_equalization)
        hist_menu.add_command(label="Kontrast Germe", command=self.apply_contrast_stretching)
        thresh_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Eşikleме", menu=thresh_menu)
        thresh_menu.add_command(label="Manuel Eşikleme", command=self.apply_manual_threshold)
        thresh_menu.add_command(label="OTSU Metodu", command=self.apply_otsu_threshold)
        thresh_menu.add_command(label="Kapur Entropi Metodu", command=self.apply_kapur_threshold)
        morph_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="İkili Görüntü İşlemleri", menu=morph_menu)
        morph_menu.add_command(label="Genişletme (Dilation)", command=lambda: self.apply_morphology('dilation'))
        morph_menu.add_command(label="Aşındırma (Erosion)", command=lambda: self.apply_morphology('erosion'))
        morph_menu.add_command(label="Ağırlık Merkezi Bul", command=self.calculate_centroid)
        morph_menu.add_command(label="İskelet Çıkar", command=self.extract_skeleton)
    
    # --- Dosya ve Görüntü Yönetimi ---
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Bir Görüntü Dosyası Seçin",
            filetypes=[("Tüm Görüntü Dosyaları", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff"), ("JPEG Dosyaları", "*.jpg;*.jpeg"), ("PNG Dosyaları", "*.png"), ("BMP Dosyaları", "*.bmp")]
        )
        if not file_path: return
        try:
            with open(file_path, 'rb') as f:
                file_bytes = np.fromfile(f, dtype=np.uint8)
            self.original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if self.original_image is None:
                raise ValueError("OpenCV dosyayı bir görüntü olarak yorumlayamadı. Dosya bozuk veya desteklenmeyen bir formatta olabilir.")
            self.processed_image = self.original_image.copy()
            self.is_binary = False
            self.display_image(self.original_image, self.original_label)
            self.update_processed_view()
        except Exception as e:
            messagebox.showerror("Dosya Okuma Hatası", f"Görüntü yüklenirken bir hata oluştu:\n\n{e}")

    def reset_image(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.is_binary = False
            self.update_processed_view()
        else:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin.")

    def update_processed_view(self):
        if self.processed_image is not None:
            self.display_image(self.processed_image, self.processed_label)
            self.update_histogram()

    def display_image(self, img_cv, label):
        if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_cv
        img_pil = Image.fromarray(img_rgb)
        label_w, label_h = label.winfo_width(), label.winfo_height()
        if label_w < 2 or label_h < 2:
             label_w, label_h = 600, 500
        img_pil.thumbnail((label_w, label_h), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        label.config(image=img_tk)
        label.image = img_tk

    def show_context_menu(self, event, image_type):
        context_menu = Menu(self.root, tearoff=0)
        context_menu.add_command(label="Farklı Kaydet", command=lambda: self.save_image(image_type))
        context_menu.tk_popup(event.x_root, event.y_root)

    # *** BU FONKSİYON GÜNCELLENDİ VE DAHA SAĞLAM HALE GETİRİLDİ ***
    def save_image(self, image_type):
        img_to_save = None
        if image_type == 'original' and self.original_image is not None:
            img_to_save = self.original_image
        elif image_type == 'processed' and self.processed_image is not None:
            img_to_save = self.processed_image

        if img_to_save is None:
            messagebox.showwarning("Uyarı", "Kaydedilecek bir görüntü bulunamadı.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG dosyası", "*.png"),
                ("JPEG dosyası", "*.jpg"),
                ("BMP dosyası", "*.bmp"),
                ("Tüm Dosyalar", "*.*")
            ]
        )

        # Kullanıcı 'İptal'e basarsa file_path boş olur, bu yüzden kontrol ediyoruz.
        if file_path:
            try:
                # cv2.imwrite başarılı olursa True, olmazsa False döndürür.
                success = cv2.imwrite(file_path, img_to_save)
                if success:
                    messagebox.showinfo("Başarılı", f"Görüntü şuraya kaydedildi:\n{file_path}")
                else:
                    messagebox.showerror(
                        "Kayıt Hatası",
                        "Görüntü dosyası diske yazılamadı.\n\n"
                        "Olası Nedenler:\n"
                        "- Dosya yolunda geçersiz karakterler.\n"
                        "- Kaydetmeye çalıştığınız klasör için yazma izniniz yok."
                    )
            except Exception as e:
                # Beklenmedik başka bir hata olursa kullanıcıyı bilgilendir.
                messagebox.showerror("Beklenmedik Hata", f"Görüntü kaydedilirken bir hata oluştu:\n\n{str(e)}")

    # --- Diğer tüm görüntü işleme fonksiyonları (değişiklik yok) ---
    def _ensure_image_loaded(self):
        if self.processed_image is None: messagebox.showwarning("Uyarı", "Lütfen önce bir görüntü yükleyin."); return False
        return True
    def _ensure_binary_image(self):
        if not self._ensure_image_loaded(): return False
        if not self.is_binary: messagebox.showwarning("Uyarı", "Bu işlem yalnızca ikili (siyah-beyaz) görüntülerde çalışır.\nLütfen önce bir eşikleme metodu uygulayın."); return False
        return True
    def _get_gray_image(self):
        if len(self.processed_image.shape) == 3: return cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        return self.processed_image.copy()
    def apply_filter_with_kernel(self, filter_type):
        if not self._ensure_image_loaded(): return
        ksize = simpledialog.askinteger("Çekirdek Boyutu", "Lütfen tek bir tamsayı girin (örn: 3, 5, 7):", minvalue=3, maxvalue=21)
        if ksize and ksize % 2 == 1:
            if filter_type == 'average': self.processed_image = cv2.blur(self.processed_image, (ksize, ksize))
            elif filter_type == 'median': self.processed_image = cv2.medianBlur(self.processed_image, ksize)
            elif filter_type == 'gaussian': self.processed_image = cv2.GaussianBlur(self.processed_image, (ksize, ksize), 0)
            self.is_binary = False; self.update_processed_view()
        elif ksize: messagebox.showerror("Hata", "Çekirdek boyutu tek sayı olmalıdır.")
    def apply_edge_detection(self):
        if not self._ensure_image_loaded(): return
        gray = self._get_gray_image(); grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3); grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x); abs_grad_y = cv2.convertScaleAbs(grad_y)
        self.processed_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        self.is_binary = False; self.update_processed_view()
    def apply_sharpen(self):
        if not self._ensure_image_loaded(): return
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        self.processed_image = cv2.filter2D(self.processed_image, -1, kernel)
        self.is_binary = False; self.update_processed_view()
    def apply_rotation(self):
        if not self._ensure_image_loaded(): return
        angle = simpledialog.askfloat("Döndürme Açısı", "Açıyı girin (derece):", minvalue=-360, maxvalue=360)
        if angle is not None:
            (h, w) = self.processed_image.shape[:2]; center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0); self.processed_image = cv2.warpAffine(self.processed_image, M, (w, h))
            self.update_processed_view()
    def apply_mirror(self, flip_code):
        if not self._ensure_image_loaded(): return
        self.processed_image = cv2.flip(self.processed_image, flip_code); self.update_processed_view()
    def apply_shearing(self):
        if not self._ensure_image_loaded(): return
        shear_factor = simpledialog.askfloat("Shearing Faktörü", "Yatay kaydırma faktörünü girin (örn: 0.5):", minvalue=-2.0, maxvalue=2.0)
        if shear_factor is not None:
            (h, w) = self.processed_image.shape[:2]; M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
            new_w = int(w + h * abs(shear_factor)); self.processed_image = cv2.warpAffine(self.processed_image, M, (new_w, h))
            self.update_processed_view()
    def update_histogram(self):
        self.hist_ax.clear(); img = self.processed_image
        if img is None: self.canvas.draw(); return
        if len(img.shape) == 3:
            color = ('b', 'g', 'r'); [self.hist_ax.plot(cv2.calcHist([img], [i], None, [256], [0, 256]), color=col) for i, col in enumerate(color)]
        else: self.hist_ax.plot(cv2.calcHist([img], [0], None, [256], [0, 256]), color='black')
        self.hist_ax.set_title("Histogram"); self.hist_ax.set_xlabel("Piksel Değeri"); self.hist_ax.set_ylabel("Piksel Sayısı"); self.fig.tight_layout(); self.canvas.draw()
    def apply_histogram_equalization(self):
        if not self._ensure_image_loaded(): return
        self.processed_image = cv2.equalizeHist(self._get_gray_image()); self.is_binary = False; self.update_processed_view()
    def apply_contrast_stretching(self):
        if not self._ensure_image_loaded(): return
        if len(self.processed_image.shape) == 3:
             b, g, r = cv2.split(self.processed_image)
             b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX); g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX); r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)
             self.processed_image = cv2.merge((b, g, r))
        else: self.processed_image = cv2.normalize(self.processed_image, None, 0, 255, cv2.NORM_MINMAX)
        self.is_binary = False; self.update_processed_view()
    def apply_manual_threshold(self):
        if not self._ensure_image_loaded(): return
        val = simpledialog.askinteger("Eşik Değeri", "Eşik değeri girin (0-255):", minvalue=0, maxvalue=255)
        if val is not None: _, self.processed_image = cv2.threshold(self._get_gray_image(), val, 255, cv2.THRESH_BINARY); self.is_binary = True; self.update_processed_view()
    def apply_otsu_threshold(self):
        if not self._ensure_image_loaded(): return
        thresh_val, self.processed_image = cv2.threshold(self._get_gray_image(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU); self.is_binary = True
        messagebox.showinfo("OTSU Metodu", f"Hesaplanan en iyi eşik değeri: {int(thresh_val)}"); self.update_processed_view()
    def apply_kapur_threshold(self):
        if not self._ensure_image_loaded(): return
        gray = self._get_gray_image(); hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel(); hist_prob = hist / hist.sum()
        max_entropy = -np.inf; best_thresh = 0
        for t in range(1, 256):
            p_obj = np.sum(hist_prob[:t]); p_bck = np.sum(hist_prob[t:])
            if p_obj == 0 or p_bck == 0: continue
            h_obj_sum = 0
            for p in hist_prob[:t]:
                if p > 0: h_obj_sum += (p / p_obj) * np.log(p / p_obj)
            h_obj = -h_obj_sum; h_bck_sum = 0
            for p in hist_prob[t:]:
                if p > 0: h_bck_sum += (p / p_bck) * np.log(p / p_bck)
            h_bck = -h_bck_sum; total_entropy = h_obj + h_bck
            if total_entropy > max_entropy: max_entropy = total_entropy; best_thresh = t
        _, self.processed_image = cv2.threshold(gray, best_thresh, 255, cv2.THRESH_BINARY); self.is_binary = True
        messagebox.showinfo("Kapur Entropi", f"Hesaplanan en iyi eşik değeri: {best_thresh}"); self.update_processed_view()
    def apply_morphology(self, op_type):
        if not self._ensure_binary_image(): return
        ksize = simpledialog.askinteger("Çekirdek Boyutu", "Çekirdek boyutu girin (örn: 3, 5):", minvalue=3, maxvalue=21)
        if ksize and ksize % 2 == 1:
            kernel = np.ones((ksize, ksize), np.uint8)
            if op_type == 'dilation': self.processed_image = cv2.dilate(self.processed_image, kernel, iterations=1)
            elif op_type == 'erosion': self.processed_image = cv2.erode(self.processed_image, kernel, iterations=1)
            self.update_processed_view()
        elif ksize: messagebox.showerror("Hata", "Çekirdek boyutu tek sayı olmalıdır.")
    def calculate_centroid(self):
        if not self._ensure_binary_image(): return
        contours, _ = cv2.findContours(self.processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: messagebox.showwarning("Bulunamadı", "Görüntüde bir nesne (kontur) bulunamadı."); return
        img_with_centroid = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
        for i, c in enumerate(contours):
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"]); cY = int(M["m01"] / M["m00"])
                cv2.drawContours(img_with_centroid, [c], -1, (0, 255, 0), 2); cv2.circle(img_with_centroid, (cX, cY), 5, (0, 0, 255), -1)
        c_largest = max(contours, key=cv2.contourArea); M_largest = cv2.moments(c_largest)
        if M_largest["m00"] != 0:
            cX_l = int(M_largest["m10"] / M_largest["m00"]); cY_l = int(M_largest["m01"] / M_largest["m00"])
            messagebox.showinfo("Ağırlık Merkezi", f"Görüntüde {len(contours)} nesne bulundu.\nEn büyüğünün ağırlık merkezi: ({cX_l}, {cY_l})")
        self.processed_image = img_with_centroid; self.is_binary = False; self.update_processed_view()
    def extract_skeleton(self):
        if not self._ensure_binary_image(): return
        img_norm = self.processed_image.copy(); img_norm[img_norm == 255] = 1
        skeleton = skeletonize(img_norm)
        self.processed_image = (skeleton * 255).astype(np.uint8); self.is_binary = True; self.update_processed_view()

# Uygulamayı Başlat
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
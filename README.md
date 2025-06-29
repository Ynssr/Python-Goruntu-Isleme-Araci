# Python Görüntü İşleme Aracı

Bu proje, Python kullanılarak geliştirilmiş, temel ve orta düzey görüntü işleme operasyonlarını kullanıcı dostu bir arayüz üzerinden gerçekleştirebilen bir masaüstü uygulamasıdır. Uygulama, `Tkinter`, `OpenCV`, `Matplotlib` ve `Scikit-image` gibi popüler kütüphanelerden faydalanmaktadır.

Kullanıcılar, bir görüntüyü yükleyebilir, üzerinde çeşitli filtreler, geometrik dönüşümler ve histogram işlemleri uygulayabilir ve sonuçları anlık olarak hem görüntü hem de histogram grafiği üzerinde gözlemleyebilirler.

## Özellikler

### 1. Temel Dosya ve Görüntü İşlemleri

  - Görüntü Açma: `.bmp`, `.jpg`, `.png` gibi farklı formatlardaki görüntüleri açma.
  - Anlık Görüntüleme:** Orijinal ve işlenmiş görüntüyü yan yana gösterme.
  - Kaydetme: Orijinal veya işlenmiş görüntüyü istenilen formatta (`.png`, `.jpg`, `.bmp`) kaydetmek için sağ tıklama menüsü.
  - Sıfırlama: Yapılan tüm işlemleri geri alıp görüntüyü orijinal haline döndürme.

### 2. Görüntü Filtreleri

  - Kenar Bulma: Sobel filtresi ile kenarları tespit etme.
  - Bulanıklaştırma: Ortalama (Average) ve Gauss (Smoothing) filtreleri.
  - Gürültü Giderme: Ortanca (Median) filtresi.
  - Keskinleştirme: Görüntü detaylarını belirginleştiren keskinleştirme filtresi.

### 3\. Geometrik Dönüşümler

  - Döndürme: Görüntüyü istenilen açıda döndürme.
  - Aynalama: Görüntüyü yatay veya dikey olarak yansıtma.
  - Shearing (Kaydırma): Görüntüye eğme/kaydırma efekti uygulama.

### 4\. Histogram İşlemleri

  - Dinamik Histogram: İşlenmiş görüntünün histogramını (gri seviye veya R-G-B kanalları için) anlık olarak grafik üzerinde gösterme.
  - Histogram Eşitleme (Equalization): Görüntü kontrastını otomatik olarak iyileştirme.
  - Kontrast Germe (Stretching): Piksel değerlerini tüm aralığa yayarak kontrastı manuel olarak artırma.

### 5. Görüntü Eşikleme (Thresholding)

  - Manuel Eşikleme: Kullanıcının belirlediği bir eşik değerine göre görüntüyü ikili (siyah-beyaz) formata çevirme.
  - OTSU Metodu: Görüntü histogramını analiz ederek en uygun eşik değerini otomatik bulma.
  - Kapur Entropi Metodu:** Entropi maksimizasyonuna dayalı otomatik eşikleme.

### 6. İkili Görüntü Operasyonları

  - Genişletme (Dilation): İkili görüntüdeki beyaz bölgeleri genişletme.
  - Aşındırma (Erosion): İkili görüntüdeki beyaz bölgeleri daraltma.
  - Ağırlık Merkezi Bulma: Görüntüdeki nesnelerin ağırlık merkezini (centroid) hesaplayıp gösterme.
  - İskelet Çıkarma: Nesnenin topolojik yapısını koruyan tek piksellik iskeletini elde etme.

## Kullanılan Teknolojiler

  - Python 3.x**
  - Tkinter: Grafiksel kullanıcı arayüzü (GUI) için.
  - OpenCV-Python (`cv2`): Ana görüntü işleme operasyonları için.
  - Numpy: Matris ve sayısal işlemler için.
  - Matplotlib: Histogram grafiğini çizdirmek ve arayüze entegre etmek için.
  - Scikit-image (`skimage`): İskelet çıkarma gibi özel algoritmalar için.
  - Pillow (`PIL`): Görüntüleri Tkinter arayüzünde göstermek için.

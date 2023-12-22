# FaceSwap
*get_landmarks fonksiyonu*
<br>
<br>
<img width="316" alt="Ekran görüntüsü 2023-12-22 134159" src="https://github.com/mertefeyildiz/FaceSwap/assets/67926547/71b2b4c3-1b5f-496d-891e-a2ac080b7355">

<br>
Bu fonksiyon, bir görüntüden yüz tespiti yaparak tespit edilen yüzün 68 adet landmark (önemli nokta) koordinatlarını döndürmektedir. Bu işlem, Dlib kütüphanesinin yüz tespit ve landmark çıkarma yeteneklerini kullanır.

İşte bu fonksiyonun adım adım açıklaması:

DETECTOR ve PREDICTOR İlk Ayarlar:

<details>
<summary>DETECTOR</summary>

```python
rects = DETECTOR(im, 1)
```
DETECTOR değişkeni, Dlib kütüphanesinin içinde bulunan yüz tespit (face detection) modelini temsil eder.
DETECTOR(im, 1) komutu, görüntü üzerinde yüzleri tespit eder. İkinci parametre olan 1, tespit edilen yüzleri upsample etme işlemini ifade eder, yani daha hassas bir tespit için görüntüyü büyütmeye olanak tanır.

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
<summary>get_landmarks Fonksiyonu</summary>

```python
rects = DETECTOR(im, 1)
def get_landmarks(im):
    # ...
    return landmarks
```


rects = DETECTOR(im, 1)
DETECTOR değişkeni, Dlib kütüphanesinin içinde bulunan yüz tespit (face detection) modelini temsil eder.
DETECTOR(im, 1) komutu, görüntü üzerinde yüzleri tespit eder. İkinci parametre olan 1, tespit edilen yüzleri upsample etme işlemini ifade eder, yani daha hassas bir tespit için görüntüyü büyütmeye olanak tanır.
Yüz Sayısı Kontrolü:

python
Copy code
if len(rects) > 1:
    raise Exception('Too Many Faces')
if len(rects) == 0:
    raise Exception('Not Enough Faces')
rects değişkeni, tespit edilen yüzlerin dikdörtgen bölgelerini içerir.
Eğer tespit edilen yüz sayısı birden fazla ise (len(rects) > 1), bir hata fırlatılır ve "Too Many Faces" mesajı gösterilir.
Eğer hiç yüz tespit edilemezse (len(rects) == 0), yine bir hata fırlatılır ve "Not Enough Faces" mesajı gösterilir.
Landmark Koordinatlarının Çıkartılması:

python
Copy code
return numpy.array([[p.x, p.y] for p in PREDICTOR(im, rects[0]).parts()])
PREDICTOR değişkeni, Dlib kütüphanesinin yüz landmark çıkarma modelini temsil eder.
PREDICTOR(im, rects[0]) komutu, tespit edilen ilk yüz üzerinde landmark çıkarma işlemini gerçekleştirir.
p.x ve p.y, her bir landmark noktasının x ve y koordinatlarını temsil eder.
Bu koordinatlar, bir Numpy dizisi içinde saklanarak fonksiyon tarafından döndürülür.
Bu fonksiyon, bir görüntüde yüz tespiti yapar ve tespit edilen yüzün landmark koordinatlarını içeren bir Numpy dizisi döndürür. Bu landmark noktaları, yüzün çeşitli bölgelerini (gözler, burun, ağız, vb.) temsil eder ve genellikle yüzün şeklini ve özelliklerini yakalamak için kullanılır.

# FaceSwap Algoritması

<br>
<br>

![face_swap diagram](https://github.com/mertefeyildiz/FaceSwap/assets/67926547/7576273d-3e6f-4c76-85a8-196b2a31c08c)


<br>




# 1-Yüz Landmark Tespiti (get_landmarks):

Dlib kütüphanesinin yüz tespit (DETECTOR) ve yüz landmark çıkarma (PREDICTOR) modelleri kullanılarak bir görüntüdeki yüz tespit edilir.
Eğer tespit edilen yüz sayısı 1'den fazla veya hiç yoksa hata mesajı verilir.
Yüzdeki önemli noktaların (landmark) koordinatları hesaplanır ve bir Numpy dizisi olarak döndürülür.

# 2-Affine Dönüşüm Matrisi Oluşturma (transformation_from_points):

İki farklı nokta kümesi arasındaki en iyi uyumu sağlayan bir affine dönüşüm matrisi hesaplanır. Bu, Procrustes problemini çözerek yapılır.
İlk olarak, noktaların merkezi alınır, standart sapmaları normalize edilir ve ardından SVD (Tekil Değer Ayrışımı) kullanılarak dönüşüm matrisi elde edilir.
Böylece iki özellik noktalarının üst üste gelmesi sağlanır.

# 3-Maske Oluşturma (create_mask):

Yüzün göz, burun, ağız bölgelerini temsil eden gruplar tanımlanır.
Bu gruplara ait bölgeler, vücut üzerinde convex hull kullanılarak bir maske oluşturulur.
Oluşturulan maske, belirli bir ölçüde genişletilir (dilate) ve ardından bir Gauss bulanıklığı uygulanarak yumuşatılır (blur).
Maske oluşturulduktan sonra dilate ve blur işlemleri, maskeyi daha iyi ayarlamak, konturları yumuşatmak ve genel olarak daha doğru bir hale getirmek amacıyla yapılır. 

# 4-Renk Düzeltme (correct_colours):

Yüz ve vücut görüntülerine belirli bir bulanıklık uygulanarak yumuşatılır.
Yumuşatılmış vücut görüntüsü ile orijinal yüz görüntüsü toplanır, aynı zamanda orijinal yüz görüntüsünden yumuşatılmış yüz görüntüsü çıkarılır. Bu işlem, renk uyumunu artırmaya yardımcı olur.

# Örnek Çıktılar:
![face_swap diagram (1)](https://github.com/mertefeyildiz/FaceSwap/assets/67926547/bac861f3-4816-4600-8b9d-fdd0aa9dd495)
![face_swap diagram (2)](https://github.com/mertefeyildiz/FaceSwap/assets/67926547/ff5a4fee-05ca-4a18-9acb-8bc31b6fca5b)
![face_swap diagram (4)](https://github.com/mertefeyildiz/FaceSwap/assets/67926547/87566fd7-231e-4da1-91e6-4d75136991cf)
![face_swap diagram (3)](https://github.com/mertefeyildiz/FaceSwap/assets/67926547/71822eea-3582-4672-a637-77faca5c8516)




<details>
<summary>def get_landmarks</summary>
    
<img width="316" alt="Ekran görüntüsü 2023-12-22 134159" src="https://github.com/mertefeyildiz/FaceSwap/assets/67926547/71b2b4c3-1b5f-496d-891e-a2ac080b7355">
<br>

```python
rects = DETECTOR(im, 1)
```
DETECTOR değişkeni, Dlib kütüphanesinin içinde bulunan yüz tespit (face detection) modelini temsil eder.
DETECTOR(im, 1) komutu, görüntü üzerinde yüzleri tespit eder. İkinci parametre olan 1, tespit edilen yüzleri upsample etme işlemini ifade eder, yani daha hassas bir tespit için görüntüyü büyütmeye olanak tanır.

```python
if len(rects) > 1:
    raise Exception('Too Many Faces')
if len(rects) == 0:
    raise Exception('Not Enough Faces')
```
rects değişkeni, tespit edilen yüzlerin dikdörtgen bölgelerini içerir.
Eğer tespit edilen yüz sayısı birden fazla ise (len(rects) > 1), bir hata fırlatılır ve "Too Many Faces" mesajı gösterilir.
Eğer hiç yüz tespit edilemezse (len(rects) == 0), yine bir hata fırlatılır ve "Not Enough Faces" mesajı gösterilir.

```python
return numpy.array([[p.x, p.y] for p in PREDICTOR(im, rects[0]).parts()])
```

PREDICTOR değişkeni, Dlib kütüphanesinin yüz landmark çıkarma modelini temsil eder.
PREDICTOR(im, rects[0]) komutu, tespit edilen ilk yüz üzerinde landmark çıkarma işlemini gerçekleştirir.
p.x ve p.y, her bir landmark noktasının x ve y koordinatlarını temsil eder.
Bu koordinatlar, bir Numpy dizisi içinde saklanarak fonksiyon tarafından döndürülür.
Bu fonksiyon, bir görüntüde yüz tespiti yapar ve tespit edilen yüzün landmark koordinatlarını içeren bir Numpy dizisi döndürür. Bu landmark noktaları, yüzün çeşitli bölgelerini (gözler, burun, ağız, vb.) temsil eder ve genellikle yüzün şeklini ve özelliklerini yakalamak için kullanılır.



</details>

<details>
<summary>def transformation_from_points</summary>
    
Bu fonksiyon, Procrustes problemini çözmek için kullanılır ve iki nokta kümesi arasındaki en iyi uyumu bulur. Fonksiyon açıklaması:

Parametreler:

points1: İlk nokta kümesi. Buradaki noktalar, dönüşümü almak istediğimiz orijinal noktalardır.
points2: İkinci nokta kümesi. Bu noktalar, orijinal noktaların yerine geçecek olan noktalardır.
Veri Tipi Dönüşümleri:

points1 ve points2 Numpy dizilerine dönüştürülür. Bu, daha sonra kullanılacak matematiksel işlemleri gerçekleştirmek için gerekli olan veri tipini sağlar.
```python
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)
```
Merkezleme:

c1 ve c2, her bir nokta kümesinin merkezini temsil eden vektörlerdir. Bu merkez vektörleri, noktaların etrafında dönmek ve ölçeklendirmek için kullanılacaktır.
Her iki nokta kümesi de kendi merkezinden çıkartılır, böylece her iki küme de orijin etrafında hizalanır.
```python
    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
```
Ölçeklendirme:

s1 ve s2, her bir nokta kümesinin standart sapmasını temsil eden ölçek faktörleridir.
Her iki küme, kendi standart sapmasına bölünerek normalize edilir.
```python
    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2
```
SVD (Singular Value Decomposition):

Singular Value Decomposition (Tekil Değer Ayrışımı) işlemi gerçekleştirilir. Bu işlem, matris çarpanlarına ayrıştırma işlemidir.
U, S, ve Vt, SVD işleminden elde edilen bileşenlerdir.
```python
    U, S, Vt = numpy.linalg.svd(points1.T @ points2) # @ --> *
```

Döndürme Matrisi (R) Bulma:
```python
    R = (U @ Vt).T # @ --> * 
```
R, döndürme matrisidir ve SVD bileşenleri kullanılarak hesaplanır.
İki matris çarpımından elde edilen çözüm aslında U * Vt matrisidir. Ancak, bu çözümün transpozu (T) alınmalıdır.<br>
Affine Dönüşüm Matrisini Oluşturma:

numpy.hstack kullanılarak R matrisi ve translasyon vektörü birleştirilir ve sonuç olarak affine dönüşüm matrisi elde edilir.
Translasyon vektörü, (c2.T - (s2 / s1) * R @ c1.T)[:,None] ifadesi ile hesaplanır.<br>
Sonuç:

Oluşturulan affine dönüşüm matrisi, [s * R | T] formülüne uyan bir matristir ve bu matris fonksiyon tarafından döndürülür.
Bu adımlar, iki nokta kümesi arasındaki en iyi uyumu sağlayan bir affine dönüşüm matrisini oluşturmak için kullanılır.
</details>
<details>
<summary>def create_mask</summary>
Adım 1: Landmark gruplarını tanımla. Bu gruplar, yüzün göz, burun, ağız bölgelerini temsil eden landmark noktalarını içerir.

Adım 2: Boş bir maske dizisi oluştur. Bu dizide, son maskeyi saklayacağız.

Adım 3: Her bir landmark grubu için işlem yap. Bu, yüzün farklı bölgelerini kapsayan farklı maskeleri oluşturmak anlamına gelir.

Adım 4: Her bir grup içindeki landmark noktalarını al. Bu, her bir landmark grubunu oluşturan noktaların konumlarını içerir.

Adım 5: Convex hull kullanarak landmark noktalarını saran çokgeni oluştur. Bu, landmark noktalarının en dış noktalarını birleştiren bir çizgidir.

Adım 6: Convex hull içini doldurarak maskeyi oluştur. Bu, convex hull içinde kalan bölgeyi beyaz renk ile doldurarak maskeyi oluşturur.

Adım 7: Yüz maskesini yumuşatmak için bir 'feather' uygula. Bu, maskeyi genişletmek ve daha yumuşak bir geçiş elde etmek için bir işlemdir.

Adım 8: Maskeyi genişlet (dilate) ve ardından bir Gauss filtresi uygula (blur). Bu, maskeyi daha da yumuşatır ve son maskeyi elde ederiz.

```python
def create_mask(points, shape, face_scale):
    # Landmark gruplarını tanımla
    groups = [
        [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
        [27, 28, 29, 30, 31, 32, 33, 34, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
    ]

    # Boş bir maske dizisi oluştur
    mask_im = numpy.zeros(shape, dtype=numpy.float64)

    # Her bir landmark grubu için işlem yap
    for group in groups:
        # Grup içindeki landmark noktalarının konumlarını al
        landmarks = [points[idx] for idx in group]

        # Convex hull kullanarak landmark noktalarını saran çokgeni oluştur
        hull = cv2.convexHull(numpy.array(landmarks))

        # Convex hull içini doldurarak maskeyi oluştur
        cv2.fillConvexPoly(mask_im, hull, color=(1, 1, 1))

    # Yüz maskesini yumuşatmak için bir 'feather' uygula
    feather_amount = int(0.2 * face_scale * 0.5) * 2 + 1
    kernel_size = (feather_amount, feather_amount)

    # Maskeyi genişlet (dilate) ve ardından bir Gauss filtresi uygula (blur)
    mask_im = (cv2.GaussianBlur(mask_im, kernel_size, 0) > 0) * 1.0

    return mask_im

```
</details>
<details>
<summary>def correct_colours</summary>
Adım 1: Bulanıklık miktarını hesapla. Bu miktar, belirli bir oranla yüz ölçeği (face_scale) ile çarpılır ve en yakın tek sayıya yuvarlanır.

Adım 2: Bulanıklık miktarına göre bir Gauss filtresi çekirdeği oluştur. Bu çekirdek, daha sonra görüntüleri yumuşatmak için kullanılacaktır.

Adım 3: Yüz ve vücut görüntülerini belirtilen bulanıklık miktarıyla yumuşat. Bu işlem, görüntülerdeki küçük detayları azaltarak renk uyumunu artırır.

Adım 4: Renk düzeltme işlemi. Yumuşatılmış vücut görüntüsü ile orijinal yüz görüntüsünü topla, aynı zamanda orijinal yüz görüntüsünden yumuşatılmış yüz görüntüsünü çıkar. Bu işlem, yüz ve vücut renklerini uyumlu hale getirmeye yardımcı olur.

Adım 5: Sonucu 0 ile 255 arasındaki değerlerle sınırla. Bu, görüntü piksellerinin geçerli değer aralığını korumak için yapılır.
```python
def correct_colours(warped_face_im, body_im, face_scale):
    blur_amount = int(3 * 0.5 * face_scale) * 2 + 1
    kernel_size = (blur_amount, blur_amount)

    face_im_blur = cv2.GaussianBlur(warped_face_im, kernel_size, 0)
    body_im_blur = cv2.GaussianBlur(body_im, kernel_size, 0)

    return numpy.clip(0. + body_im_blur + warped_face_im - face_im_blur, 0, 255)

```
</details>

# GAN  model kullanılsaydı nasıl çalışırdı?

<br>

![face_swap diagram (5)](https://github.com/mertefeyildiz/FaceSwap/assets/67926547/7c6b07a7-5150-4d43-afdd-a38eadb86db7)


Jeneratör (Generator):

Face swapping için jeneratör, bir yüzü diğerine dönüştürmekle görevlidir. Genellikle, jeneratörün girişi, değiştirilmek istenen yüzün landmark noktalarını içeren bir vektördür. Bu vektör, encoder tarafından üretilir.
Jeneratör, bu giriş vektörünü kullanarak bir dizi ters evrişim katmanı ile yüz özelliklerini başka bir yüzde nasıl ifade edeceğini öğrenir.
Diskriminator (Discriminator):

Diskriminator, orijinal yüz ve swap edilmiş yüz arasındaki farkı öğrenmeye çalışır. Yani, gerçek bir yüz ve sahte (swap edilmiş) bir yüz arasındaki ayırt edici özellikleri öğrenir.
Diskriminator, bu öğrenmeleri kullanarak, eğitildiği veri setinde doğru bir ayrım yapabilme yeteneği kazanır.
Encoder (Landmark Detection ve Feature Extraction):

Encoder, yüzleri temsil eden landmark noktalarını ve belki de yüz özelliklerini düşük boyutlu bir latent uzaya dönüştürür. Bu, yüzleri temsil eden daha anlamlı ve öğrenilebilir bir gösterim elde etmek için kullanılır.
Encoder ayrıca, diğer yüzün landmark noktalarını bu latent uzaya dönüştürür.
Decoder (Swap Edilmiş Yüzün Oluşturulması):

Decoder, encoder tarafından elde edilen latent uzaydaki temsil ve swap edilecek yüzün landmark noktalarını kullanarak, orijinal yüzden diğerine geçişi sağlayan bir görüntü oluşturur.
Decoder, jeneratörün işleyebileceği bir forma dönüştürülmüş swap edilmiş yüzü üretir.
Eğitim:

Eğitim sırasında, jeneratör ve diskriminator birbirleriyle rekabet ederler. Jeneratör, diskriminatörü kandırmaya ve swap edilmiş yüzü gerçekmiş gibi göstermeye çalışırken, diskriminator gerçek ile sahte arasındaki farkı belirleme görevini yerine getirir.
Encoder ve decoder da, latent uzayda daha anlamlı temsiller oluşturmak için eğitilir.
Sonuç:

Eğitim tamamlandığında, jeneratör ve diskriminator, yüzleri başarılı bir şekilde swap etmek için işbirliği yapabilirler. Swap edilmiş yüz, orijinal yüze benzer bir şekilde oluşturulur ve eğitim verisinde gördükleri yüzlerden öğrendikleri özellikleri kullanarak gerçekçi bir sonuç elde edilir.






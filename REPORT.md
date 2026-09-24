# Nəticələrin təhlili

Hər iki model CPU-da, eyni bölgü və 12 epoch ilə öyrədilib. Train/validation/test ölçüləri: 54 001 / 5 999 / 10 000. Model seçimi test nəticəsinə deyil, validation loss-a əsaslanır.

| Model | Seçilən epoch | Test accuracy | Test loss |
|---|---:|---:|---:|
| CNN | 8 | 99.02% | 0.0297 |
| MLP | 9 | 97.92% | 0.0708 |

CNN 1.10 faiz bəndi daha yaxşı nəticə göstərib. CNN yaxın piksellərin əlaqəsini və eyni filtrləri şəklin müxtəlif yerlərində istifadə edir. MLP isə şəkli bir vektora çevirir. Bu, müşahidə edilən fərqin mümkün izahıdır. Nəticə bir seed ilə ölçülüb; bütün təcrübələrdə eyni fərq gözləmək olmaz.

## Təlim qrafikləri

![Loss və accuracy](artifacts/training_curves.png)

Son epoch-da CNN train accuracy 99.73%, validation accuracy 98.92% olub. MLP-də bu göstəricilər 99.58% və 97.50%-dir. MLP-nin train/validation fərqi daha böyükdür. Son epoch-larda train nəticəsi yüksək qalsa da validation loss artır; bu, overfitting əlamətidir. Ona görə ən son çəkilər əvəzinə ən aşağı validation loss olan çəkilər seçilib.

## Qarışıqlıq matrisləri

![CNN](artifacts/cnn_confusion.png)

Sətirlər həqiqi rəqəmi, sütunlar proqnozu göstərir. CNN ən çox **2 və 7** cütünü qarışdırıb: 2 → 7 istiqamətində 8, 7 → 2 istiqamətində 3 səhv; cəmi **11**. Cüt hər iki istiqamətin diaqonaldan kənar cəmi ilə seçilib. Mümkün səbəb: maili xəttin oxşarlığı və 2-nin aşağı hissəsinin zəif yazılması onu 7-yə bənzədə bilər.

![MLP](artifacts/mlp_confusion.png)

MLP-də ən çox qarışan cüt **4 və 9** olub: müvafiq istiqamətlərdə 7 və 10 səhv, cəmi 17. Qapalı üst hissə və oxşar şaquli xətt bu qarışıqlığı izah edə bilər.

## MNIST-dən kənar 7 real əl yazısı

Eyni saxlanmış CNN ilə iki şəkil dəsti yoxlanılıb. İlk nazik xətli nümunələrdə 5/7 nəticə alındıqdan sonra rəqəmlər daha qalın xətlə yenidən çəkilib. Hər iki dəstin bütün şəkilləri saxlanılıb; bunlarla model öyrədilməyib.

| Şəkil | Həqiqi rəqəm | İlk proqnoz | Yeni proqnoz |
|---|---:|---:|---:|
| picture_1.png | 3 | 5 | 3 |
| picture_2.png | 8 | 1 | 8 |
| picture_3.png | 1 | 1 | 1 |
| picture_4.png | 4 | 4 | 4 |
| picture_5.png | 5 | 5 | 5 |
| picture_6.png | 7 | 7 | 7 |
| picture_7.png | 7 | 7 | 7 |

İlk nəticə **5/7 (71.43%)**, yeni nəticə **7/7 (100%)**-dir. İlk dəstdə 3 → 5 və 8 → 1 səhvləri olub. Yeni dəstdə yeddi rəqəmin hamısı düzgün tanınıb.

![Yeni şəkillərin preprocessing addımları](artifacts/custom_preprocessing.png)

Nazik xətlər kiçildiləndə zəifləyə bilər; daha qalın xətlər modelin formanı ayırmasını asanlaşdıra bilər. Bununla yanaşı yeni rəqəmlərin forması da dəyişib. Buna görə yaxşılaşmanı yalnız qalınlığın təsiri kimi təqdim etmək olmaz.

Yeni şəkillər ilkin nəticələr görüldükdən sonra çəkilib. Bu, ayrıca kor test deyil, girişin yazılışına həssaslığı göstərən təkrar sınaqdır. Yeddi nümunədə 100% nəticə bütün real əl yazılarında 100% dəqiqlik demək deyil.

İlk şəkillər, onların manifesti, 5/7 nəticə JSON-u və qrafiki `artifacts/previous_handwriting/` daxilindədir. Cari `custom_images/` qovluğunda yeni yeddi PNG var.

CNN və MLP yenidən MNIST üzərində öyrədilərək hesabatdakı test göstəriciləri təkrar əldə olunub. Son şəkil dəyişiklikləri zamanı model çəkiləri dəyişdirilməyib. Gələcək təlim nəticələri fərqlənərsə, hesabat da yenilənməlidir.

## Bonus: öyrənilmiş filtrlər

![İlk convolution qatının filtrləri](artifacts/filters.png)

Şəkildə ilk qatın 16 ədəd 3 × 3 filtri göstərilib. Hamısında eyni rəng şkalası istifadə olunur: qırmızı müsbət, mavi mənfi çəkiləri göstərir. Filtrlər lokal piksel fərqlərinə cavab verir; ayrı-ayrılıqda tam rəqəm təsviri deyil.

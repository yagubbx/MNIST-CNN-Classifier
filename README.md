# MNIST: CNN və MLP müqayisəsi

Yagub Khalilli — Devlab, Week 2

Layihədə PyTorch ilə sıfırdan CNN və MLP öyrədilir. Məqsəd eyni təlim şərtlərində nəticələri müqayisə etmək və CNN-i MNIST-dən ayrı 7 real əl yazısında yoxlamaqdır. Hazır çəkilər və ölçülmüş nəticələr repozitoriyadadır.

## İşə salmaq

Python 3.11 istifadə olunub. VS Code-da bu qovluğu açıb terminalda yaz:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Tam təcrübəni yenidən icra etmək üçün:

```powershell
.\.venv\Scripts\python train.py
.\.venv\Scripts\python external_test.py
```

`train.py` hər iki modeli 12 epoch öyrədir, çəkiləri, loss/accuracy qrafiklərini, test nəticələrini və confusion matrix-ləri `artifacts/` qovluğuna yazır. İlk təlimdə MNIST internetdən yüklənir. Təkrar təlim həmin nəticə fayllarını yeniləyir; `REPORT.md` saxlanmış nəticələrin təhlilidir və yeni təlimdən sonra yenidən nəzərdən keçirilməlidir.

Yalnız mövcud modelin 7 real şəkil üzərində sınağı üçün `external_test.py` işlətmək kifayətdir. Bu skript təlim aparmır və internet tələb etmir.

Bir şəkli ayrıca yoxlamaq və preprocessing addımlarını saxlamaq üçün:

```powershell
.\.venv\Scripts\python predict.py custom_images/picture_3.png --stages artifacts/preprocessing
```

## Məlumat bölgüsü və ədalətli müqayisə

- Loader: `torchvision.datasets.MNIST`.
- Train: 54 001, validation: 5 999, test: 10 000 şəkil.
- Validation hər sinifdən təxminən 10% ayrılmaqla yaradılır. Rəsmi test dəsti validation kimi istifadə edilmir.
- Piksel qiymətləri float32 tipinə çevrilib 255-ə bölünür: `[0, 1]`.
- Hər iki model: seed 42, 12 epoch, batch size 128, Adam, learning rate 0.001, cross-entropy.
- Eyni bölgü indeksləri və hər epoch üçün eyni mini-batch sırası istifadə olunur.
- Hər epoch sonunda train/validation loss və accuracy hesablanır. Ən aşağı validation loss olan çəkilər saxlanır, sonra test dəstində qiymətləndirilir.
- Pretrained çəkilər, augmentation və transfer learning istifadə edilmir.

## CNN arxitekturası

| Qat | Parametrlər | Çıxış ölçüsü |
|---|---|---|
| Giriş | Grayscale | 1 × 28 × 28 |
| Conv2d | 1 → 16, kernel 3, padding 1 | 16 × 28 × 28 |
| ReLU | Aktivasiya | 16 × 28 × 28 |
| MaxPool2d | Kernel 2, stride 2 | 16 × 14 × 14 |
| Conv2d | 16 → 32, kernel 3, padding 1 | 32 × 14 × 14 |
| ReLU | Aktivasiya | 32 × 14 × 14 |
| MaxPool2d | Kernel 2, stride 2 | 32 × 7 × 7 |
| Flatten | — | 1568 |
| Linear | 1568 → 128 | 128 |
| ReLU | Aktivasiya | 128 |
| Linear | 128 → 10 | 10 logits |

MLP: `Flatten → Linear(784,256) → ReLU → Linear(256,64) → ReLU → Linear(64,10)`.

CNN-də 206 922, MLP-də 218 058 parametr var. Təlim şərtləri eynidir, amma hesablama xərci tam eyni deyil. Təlimdə logits cross-entropy-yə verilir; proqnozda softmax hesablanır.

## Yeddi real əl yazısı

`custom_images/` qovluğunda istifadəçinin göndərdiyi 7 əl yazısı PNG-si var. Fayllar dəyişdirilmədən `picture_1.png`–`picture_7.png` adlandırılıb. Etiketlər sıra ilə **3, 8, 1, 4, 5, 7, 7** kimi oxunub. Şəkillər MNIST-dən götürülməyib və təlimdə istifadə olunmur.

`manifest.json` faylında hər şəklin adı, həqiqi rəqəmi və SHA-256 yoxlama cəmi var. `external_test.py` bu siyahıdakı bütün şəkilləri yoxlayır; say 5 ilə məhdud deyil. Qovluğa yeni şəkil əlavə etdikdə manifestə də adı və düzgün etiketi yazılmalıdır. Skript siyahı ilə qovluq uyğun gəlmədikdə xəta verir, şəkli səssizcə buraxmır.

`predict.py`: grayscale → fonun təyini və invert → kontrast → rəqəmin kəsilməsi → nisbəti qoruyaraq resize → 28 × 28 fonda mərkəzləşdirmə → 255-ə bölmə. Addımlar `artifacts/custom_preprocessing.png` şəklində göstərilir. Orijinal şəkillərin 28 × 28 olması lazım deyil.

Sınaq saxlanmış `cnn.pt` faylını ayrıca prosesdə yükləyir, `eval()` rejimində gradient hesablamadan proqnoz verir. Şəkillərin hamısı qiymətləndirilir, səhv nəticələr də saxlanır.

İlk nazik xətli şəkillər 5/7 nəticə verib. Daha qalın xətlə yenidən çəkilmiş yeddi şəkil 7/7 nəticə verib. Əvvəlki şəkillər və nəticələr `artifacts/previous_handwriting/` qovluğunda saxlanılır. Yeni şəkillər ilk nəticələr görüldükdən sonra çəkilib; bu, müstəqil kor sınaq deyil. Yazılış forması da dəyişdiyindən fərqi yalnız xətt qalınlığına bağlamaq olmaz. Model yenidən öyrədilməyib.

## Bonuslar

İki bonus əlavə olunub: şəkil qəbul edən Flask endpoint-i və ilk convolution qatının filtrlərinin göstərilməsi.

Sadə şəkil yükləmə səhifəsi üçün:

```powershell
.\.venv\Scripts\python app.py
```

Brauzerdə http://127.0.0.1:5000 aç, şəkil seç və **Yoxla** düyməsinə bas. Hazır model istifadə olunur, yenidən təlim aparılmır. Terminalı açıq saxla; dayandırmaq üçün Ctrl+C bas. Bu lokal demo üçündür.

`POST /predict` sorğusu `image` adlı fayl qəbul edir və proqnoz rəqəmini, softmax skorunu və ilk üç nəticəni JSON qaytarır. Softmax skoru doğruluq zəmanəti deyil. Şəkil limiti 5 MB-dır.

Filtrlərin qrafikini yaratmaq üçün:

```powershell
.\.venv\Scripts\python show_filters.py
```

Nəticə `artifacts/filters.png` faylıdır. Bu bonuslar CNN–MLP təlim şərtlərini dəyişmir.

## Nəticələr və fayllar

Ölçülmüş nəticələr: CNN **99.02%**, MLP **97.92%**, yenidən çəkilmiş əl yazısı **7/7** (ilk sınaq **5/7**). Təhlil: [REPORT.md](REPORT.md).

- `models.py`: CNN və MLP qatları.
- `train.py`: məlumat bölgüsü, təlim, müqayisə və qrafiklər.
- `predict.py`: saxlanmış model və bir şəkil üçün inference.
- `external_test.py`: 7 şəkil və preprocessing qrafiki.
- `app.py`, `static/index.html`: sadə şəkil yükləmə səhifəsi və endpoint.
- `show_filters.py`: öyrənilmiş filtrlərin qrafiki.
- `artifacts/`: çəkilər, tarixçələr, bölgü indeksləri və nəticələr.
- `custom_images/`: 7 PNG və mənbə məlumatı.

GitHub-a bu qovluğun içindəkiləri yüklə; `.venv/`, `data/` və `__pycache__/` daxil edilməsin. Əsas səhifədə README görünməlidir.

## Mənbələr

- MNIST: http://yann.lecun.com/exdb/mnist/

Kodun hazırlanmasında AI köməyindən istifadə olunub.

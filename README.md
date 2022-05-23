# Anadolu Hayat Emeklilik Datathon

## Proje Yapısı

```
anadolu-hayat-emeklilik-datathon/
├─ data/
├─ eda/
├─ models/
│  ├─ lightgbm/
│  ├─ xgboost/
├─ src/
├─ .gitignore
├─ requirements.txt
├─ README.md
```

İşlenmiş ve işlenmemiş veriler `data` dizini içindedir.

Notebooklar, görselleştirmeler ve diğer veri analizleri `eda` dizini içindedir.

Modeller, modellerin sonuçlarının görselleştirmeleri ve modellerin tahminleri `models` dizini içindedir.

Python modülleri `src` dizini içindedir.

## Donanım ve Yazılımlar

```
CPU: Intel(R) Core(TM) i5-9300H CPU @ 2.40GHz
GPU: NVIDIA GeForce GTX 1050
OS: Ubuntu 20.04.4 LTS
Python: 3.9.12
```

## Kurulum

```
git clone https://github.com/gunesevitan/anadolu-hayat-emeklilik-datathon.git
cd anadolu-hayat-emeklilik-datathon
virtualenv --python=/usr/bin/python3.9 venv
source venv/bin/activate
pip install -r requirements.txt
```

`settings.py` içindeki `ROOT` değişkeni projenin absolute pathi olarak değiştirilmelidir.  

## Problem Tanımı

Bireysel emeklilik sisteminde 2020 senesi boyunca aylık katkı payı vadesi olan müşterilerin bir sonraki yılın ilk çeyreğinde katkı payı artışında (aylık ödemeleri gereken vade miktarında enflasyon oranının üstünde artışa gidilmesi) bulunup bulunmayacaklarının tahmin edilmesi. Başarı metriği olarak F skoru (F1 Score) kullanılmıştır.

## Validasyon ve Metrikler

Cross-validation olarak 5 stratified fold kullanılmıştır. Foldlar içinde target verisi stratify edilmiştir. (`validation.py`)

Hata metrikleri olarak accuracy, ROC AUC, precision, recall, specificity ve F1 skorları kullanılmıştır. (`metrics.py`)

## Veri Ön İşleme

Bütün veri ön işleme adımları `preprocessing.py` modülü içinde yapılmıştır.

* SOZLESME_KOKENI sütunu içindeki TRANS değerleri TRANS_C olarak değiştirilmiştir
* GELIR sütunu nümerik yapılmıştır ve negatif değerler 0 ile değiştirilmiştir
* SOZLESME_KOKENI_DETAY sütunu içindeki eksik veriler NEW olarak doldurulmuştur
* MUSTERI_SEGMENTI sütunu içindeki eksik veriler LightGBM classifier ile tahmin edilip doldurulmuştur
* KAPSAM_TIPI sütunu içindeki eksik veriler mod değer ile doldurulmuştur
* Test sette görünmeyen kategorisi olan kategorik değişkenler Ordinal Encoder ile diğer kategorik değişkenler Label Encoder ile labellara çevirilmiştir
* Sözleşmenin ay olarak toplam süresi BASLANGIC_TARIHI sütunundan çıkarılmıştır
* Müşterilerin yaşları 2020 - DOGUM_TARIHI olarak hesaplanmıştır
* Sadece pozitif değerleri olan sürekli değişkenlere `log(x + 1)` işlemi yapılmıştır
* Hem pozitif hem negatif değerleri olan sürekli değişkenlere `log(x + 1 - min(x))` işlemi yapılmıştır
* Ayların vade tutarları ve ödenen tutarlarından lineer ve istatistiki değişkenler yaratılmıştır
* Yüksek korelasyonu olan değişken gruplarının boyutları PCA ile azaltılmıştır
* Poliçe ve müşteri değişkenleri üzerinde KMeans algoritması ile kümeler yaratılmıştır
* POLICE_SEHIR değişkeninin kategorilerinin değerleri, sıklıkları ile değiştirilmiştir
* Aralık ayı vade tutarı ve ödenen tutar üzerinde istatistiki değişkenler yaratılmıştır
* Cross-validation döngüsü içinde target encoding yapılmıştır

## Modeller

LightGBM (`lgb_trainer.py`) ve XGBoost (`xgb_trainer.py`) binary classifierlar kullanılmıştır.
Bu modellerin parametreleri, cross-validation skorunu maksimize edecek şekilde ayarlanmıştır.
F1 skoru maksimize etmek için training esnasında validasyon ROC AUC skoru hesaplanarak o skora göre early stopping yapılmıştır.

Training `python main.py ../models/lightgbm/config.yaml` komutuyla başlatılabilir.

## Model Sonrası İşleme

Müşterilerin kategorik değişkenlerinden 2 adet CUSTOMER_ID değişkeni yaratılmıştır.
Bu değişkenler training ve test setteki aynı müşterileri bulup onların target değerinin ortalamasını yazdırmak için kullanılmıştır.

Her model 3 farklı random seed ile eğitildikten sonra tahminleri yazdırılmıştır.
0 ve 1 arasındaki tahminler toplanıp, model sayısına bölünmüştür.

0 ve 1 arasındaki tahminler, training setteki precision-recall eğrisinin argmaxı olan threshold ile labellara çevirilmişlerdir.

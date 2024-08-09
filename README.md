## Dilbazlar Ekibi Kimdir?
<img width=700px src='https://github.com/user-attachments/assets/b9976097-8bb3-46d8-8167-1f7443df5ede'> 

Dilbazlar ekibi Teknofest Türkçe Doğal Dil İşleme yarışmasında yarışmak için kurulan 4 üyeden oluşan bir takımdır. Bilgisayar mühendisi Halil İbrahim Hatun, Muhammed Ensar Çıtak, Büşra Kurun ve lise öğrencisi Ahmet Akay'dan oluşmaktadır. Ekipte herkes veri kazıma ve veri temizleme aşamasında yer almıştır. Model eğitimi kısmında takım kaptanı Halil ve ekip üyesi Ensar çalışmalarını yürütürken Büşra ve Ahmet de dokümantasyonve test aşamalarını gerçekleştirmiştir.

## Proje Nedir?
Bu proje, **DSM-5'e** dayalı olarak Anksiyete ve Depresyon bozukluklarını ele almaktadır. Projede, anksiyete ve depresyon gibi rahatsızlıklar; **Distimi, PMDD (Prementrüel Disforik Bozukluk), Agorafobi, Seçici Dilsizlik, Panik Bozukluk, Sosyal Anksiyete ve Fobi** gibi alt kategorilere ayrılmaktadır.

Depresyon ve Anksiyete etiketleri, diğer etiketlerden farklı olarak model tarafından tespit edilen sonuçlara değil, belirlenen eşik değerlere dayalı olarak seçilir. Örneğin, Distimi ve PMDD üzerine tahmin yapan bir yapay zeka modeli, çıktıların belirlenen eşik değerin altında kalması durumunda otomatik olarak "Depresyon" olarak etiketlenecektir.

## Proje Aşamaları
![Proje Aşamaları](https://github.com/user-attachments/assets/164d3075-e0c6-4ec5-96c3-d5ab0d7976aa)

1. Veri Kazıma ve Önişleme
2. Model Eğitimi
3. Test ve Optimizasyon
4. Chatbot Seneryolarının Oluşturulması

### 1. Veri Kazıma ve Önişleme Aşaması

#### 1.1. Veri Toplama
![image](https://github.com/user-attachments/assets/2a420beb-d946-403b-97ba-951945eb540b)
Veri kazıma aşamasında, **X (Twitter)** ve **Reddit** sosyal medya platformlarından etiketli veriler toplanmış ve İngilizce bir veri seti oluşturulmuştur. Ardından, bu veriler HuggingFace'den **Helsinki-NLP/opus-mt-tc-big-en-tr** modeli kullanılarak Türkçeye çevrilmiştir.

Buna ek olarak, **Youtube** yorumları ve **Ekşi Sözlük** verileri de kullanılarak **organik Türkçe** veri elde edilmiştir. Bu veriler, çevrilmiş veri seti ile birleştirilmiştir.

Elde edilen tüm veri seti, LLM modeli olan Gemini 1.5 Flash (ücretsiz versiyonu kullanılmıştır) ile augmentasyon işlemi uygulanarak genişletilmiştir.

Bu işlemler sonucunda elde edilen Türkçe veri setlerimiz:
- [**89,900** adet hasta veya normal olarak etiketlenmiş veri](https://huggingface.co/datasets/halilibr/dilbazlar-disorder-detection-tr-dataset),
- [**43,400** adet anksiyete veya normal olarak etiketlenmiş veri](https://huggingface.co/datasets/halilibr/dilbazlar-anxiety-binary-tr-dataset),
- [**57,600** adet depresyon veya normal olarak etiketlenmiş veri](https://huggingface.co/datasets/halilibr/dilbazlar-depression-binary-tr-dataset),
- [**27,600** adet distimi veya PMDD (premenstrüel disforik bozukluk) olarak etiketlenmiş veri](https://huggingface.co/datasets/halilibr/dilbazlar-depression-recognition-multilabel-augmented-cleaned-tr-dataset),
- [**15,700** adet Agorafobi, Panik, Fobi, Seçici Dilsizlik, Sosyal Anksiyete olarak etiketlenmiş veriden](https://huggingface.co/datasets/halilibr/dilbazlar-anxiety-disorders-recognition-not-augmented-not-anxiety-multilabel-tr-dataset) oluşmaktadır
  
Bu çalışma, Türkiye literatüründe bu alanda yapılan ilk çalışma olma özelliğini taşımaktadır.

Kullanılan Kod Parçaları:
- [Youtube Scraping İşlemi](https://github.com/Halil3509/TEKNOFEST-Dilbazlar/blob/main/Notebooks/Youtube_Scraping.ipynb)
- [Reddit Scraping Dosyası](https://github.com/Halil3509/TEKNOFEST-Dilbazlar/tree/main/Reddit-Scraping)
- []()
- [LLM-based Augmentation](https://github.com/Halil3509/TEKNOFEST-Dilbazlar/blob/main/Notebooks/Gemini_Augmentation.ipynb)
- [Youtube Scraping İşlemi]()

### 2. Model Eğitimi
![image](https://github.com/user-attachments/assets/0f97f8b6-7d3d-4762-8d2f-ef6e8b258b0f)


Model eğitimi sürecinde, BERT tabanlı önceden eğitilmiş modeller olan **dbmdz/bert-base-turkish-cased** ve **dbmdz/bert-base-turkish-128k-uncased** modelleri kullanılmıştır.

Model eğitiminde, ilk olarak bir içeriğin "hasta" olup olmadığını belirlemek amacıyla bir model geliştirilmiştir. Eğer hastalık tespiti yapılırsa, sistem anksiyete ve depresyon için iki ayrı ikili modele yönlendirilir. Anksiyete veya depresyon modelinin çıktısı belirlenen eşik değerini aşarsa, sistem bu hastalıklar üzerine çalışan detaylı modellere yönlendirilir.

Anksiyete tarafında Agorafobi, Fobi, Sosyal Anksiyete, Seçici Dilsizlik ve Panik Bozukluk; depresyon tarafında ise Distimi ve PMDD gibi hastalıklar detaylı modellerde ele alınmaktadır. Eğer anksiyete veya depresyon tarafında ağırlıklı sonuçlar eşik değerini aşamazsa, sonuçlar yalın olarak "Anksiyete" veya "Depresyon" olarak hesaplanır.

### 3. Test ve Optimizasyon 
![Screenshot 2024-08-09 122555](https://github.com/user-attachments/assets/b934cb77-ee39-436a-8a52-49dc088b6c5f)

<br>

Model çıktıları:
| Model Adı                 | F1 Skoru | Doğruluk (Acc) |
|---------------------------|----------|----------------|
| [Hastalık mı değil mi Modeli](https://huggingface.co/halilibr/dilbazlar-binary-disorder-detection-model-acc-98.5)| %97,4       | %98,1          |
| [Anksiyete İkili (Binary) Modeli](https://huggingface.co/halilibr/dilbazlar-anxiety-disorder-binary-detection-model-acc-98.7)     | %98,9   | %98,7          |
| [Depresyon İkili (Binary) Modeli](https://huggingface.co/halilibr/dilbazlar-depression-binary-detection-model-acc-98.3)     | %98,5     | %98,3         |
| [Anksiyete Spesifik Modeli](https://huggingface.co/halilibr/dilbazlar-bert-uncased-anxiety-disorders-recognition-balanced-tr-model-acc-92.7)     | %90,1     | %92,7          |
| [Depresyon Spesifik Modeli](https://huggingface.co/halilibr/dilbazlar-depression-disorders-recognition-tr-model-acc-84)  | %84,2     | %84          |

<br>

Test ve değerlendirme tarafında [tranformers-interpret](https://github.com/cdpierse/transformers-interpret) kütüphanesi kullanılmıştır. Bu sayede çıktıdaki kelimelerin hedef etiket ile nasıl çekinlendiği görülmektedir.

*Depresyon spesifik modelinden çıkan Agorafobi çıktısının cümleyi çekimleme şekli:*
![image](https://github.com/user-attachments/assets/1843952d-bc60-427d-862e-50ad32ea070b)

Test olarak F1 ve Doğruluk metriklerinin yanı sıra günlük hayat örnek personalarıyla ürünün uygulanabilirliği test edilmiştir. (Diğer aşamada detaylara erişebilirsiniz.)


### 4. Chatbot Seneryolarının Oluşturulması 🤖
![WhatsApp Image 2024-08-09 at 1 34 14 PM](https://github.com/user-attachments/assets/c1288884-8cc9-491c-92f4-a8bb6cb4db91)

Web arayüz olarak Streamlit kütüphanesi kullanılmıştır. Ana ekranda kullanıcı sol taraftan geçmiş konuşmalarına erişebilir veya yeni bir konuşma oluşturabilmektedir. Sol tarafta verilen grafik, 10 adet verilmiş etiketlerin girdiye bağlı olarak dağılımını canlık olarak göstermektedir. 

Chatbot olarak Gemini 1.5 flash modeli ücretsiz olarak kullanılmıştır. Kişinin hastalık bağlamlı içerik bulundurma oranı verilen eşik değerini geçiyorsa Gemini'a rahatsızlık model sonuçları ifade edilir ve gemini da bunlar düzenli bir şekilde ifade eder. 


*Chatbot'un örnek bir seneryoya verdiği çıktı*
![WhatsApp Image 2024-08-09 at 1 52 10 PM](https://github.com/user-attachments/assets/a8ea8c86-fddd-49b5-a917-df2fb2eec1d9)

## Nasıl Çalıştırılır? 🚀
Sistem streamlit kütüphanesi kullanılarak ayağı kaldırılmaktadır. 

Öncelikle yukarıda verilen linklerdeki verisetlerine erişim hakkı sağlamak zorundasınız. 

Projenin ana yoluna gidiniz. (Projeyi pycharm veya vs code gibi IDE'lerden açarsanız, konsolda otomatik olarak gelecektir.)

Sonrasında aşağıdaki kod parçasını çalıştırarak gerekli olan bütün kütüphaneleri indirin
```python
pip install -r requirements.txt
```

Bu işlemden sonra aşağıdaki kod satırlarını sırayla çalıştırarak ürünü kullanabilirsiniz. 

Not: İlk çalıştırma sırasında HuggingFace modellerinin indirilmesi zaman alacaktır.
```python
cd Chatbot

streamlit run streamlit.py
```

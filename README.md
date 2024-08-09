![Dilbazlar](https://github.com/user-attachments/assets/5d43fb9f-b59c-4222-ae53-57b89c96ab86)

## Dilbazlar Ekibi Kimdir?
Dilbazlar ekibi, İzmir Bakırçay Üniversitesi Bilgisayar Mühendisliği bölümünden Ensar Çıtak, Büşra Kurun, Halil İbrahim Hatun ve İzmir Atatürk Lisesi'nden Ahmet Akay'dan oluşmaktadır.

## Proje Nedir?
Bu proje, DSM-5'e dayalı olarak Anksiyete ve Depresyon bozukluklarını ele almaktadır. Projede, anksiyete ve depresyon gibi rahatsızlıklar; Distimi, PMDD (Prementrüel Disforik Bozukluk), Agorafobi, Seçici Dilsizlik, Panik Bozukluk, Sosyal Anksiyete ve Fobi gibi alt kategorilere ayrılmaktadır.

Depresyon ve Anksiyete etiketleri, diğer etiketlerden farklı olarak model tarafından tespit edilen sonuçlara değil, belirlenen eşik değerlere dayalı olarak seçilir. Örneğin, Distimi ve PMDD üzerine tahmin yapan bir yapay zeka modeli, çıktıların belirlenen eşik değerin altında kalması durumunda otomatik olarak "Depresyon" olarak etiketlenecektir.

## Proje Aşamaları
![Proje Aşamaları](https://github.com/user-attachments/assets/164d3075-e0c6-4ec5-96c3-d5ab0d7976aa)

1. Veri Kazıma ve Önişleme
2. Model Eğitimi
3. Test ve Optimizasyon
4. Chatbot Senaryolarının Oluşturulması

### 1. Veri Kazıma ve Önişleme Aşaması

#### 1.1. Veri Toplama
![image](https://github.com/user-attachments/assets/2a420beb-d946-403b-97ba-951945eb540b)
Veri kazıma aşamasında, X (Twitter) ve Reddit gibi sosyal medya platformlarından etiketli veriler toplanmış ve İngilizce bir veri seti oluşturulmuştur. Ardından, bu veriler HuggingFace'den **Helsinki-NLP/opus-mt-tc-big-en-tr** modeli kullanılarak Türkçeye çevrilmiştir.

Buna ek olarak, Youtube yorumları ve Ekşi Sözlük verileri de kullanılarak organik Türkçe veri elde edilmiştir. Bu veriler, çevrilmiş veri seti ile birleştirilmiştir.

Elde edilen tüm veri seti, LLM modeli olan Gemini 1.5 Flash (ücretsiz versiyonu kullanılmıştır) ile augmentasyon işlemi uygulanarak genişletilmiştir.

Bu işlemler sonucunda elde edilen Türkçe veri setleri:
- **89,900** adet hasta veya normal olarak etiketlenmiş veri,
- **43,400** adet anksiyete veya normal olarak etiketlenmiş veri,
- **57,600** adet depresyon veya normal olarak etiketlenmiş veri,
- **27,600** adet distimi veya PMDD (prementrüel disforik bozukluk) olarak etiketlenmiş veri,
- **15,700** adet Agorafobi, Panik, Fobi, Seçici Dilsizlik, Sosyal Anksiyete olarak etiketlenmiş veriden oluşmaktadır.

Bu çalışma, Türkiye literatüründe bu alanda yapılan ilk çalışma olma özelliğini taşımaktadır.

### 2. Model Eğitimi
![image](https://github.com/user-attachments/assets/0f97f8b6-7d3d-4762-8d2f-ef6e8b258b0f)


Model eğitimi sürecinde, BERT tabanlı önceden eğitilmiş modeller olan **dbmdz/bert-base-turkish-cased** ve **dbmdz/bert-base-turkish-128k-uncased** modelleri kullanılmıştır.

Model eğitiminde, ilk olarak bir içeriğin "hasta" olup olmadığını belirlemek amacıyla bir model geliştirilmiştir. Eğer hastalık tespiti yapılırsa, sistem anksiyete ve depresyon için iki ayrı ikili modele yönlendirilir. Anksiyete veya depresyon modelinin çıktısı belirlenen eşik değerini aşarsa, sistem bu hastalıklar üzerine çalışan detaylı modellere yönlendirilir.

Anksiyete tarafında Agorafobi, Fobi, Sosyal Anksiyete, Seçici Dilsizlik ve Panik Bozukluk; depresyon tarafında ise Distimi ve PMDD gibi hastalıklar detaylı modellerde ele alınmaktadır. Eğer anksiyete veya depresyon tarafında ağırlıklı sonuçlar eşik değerini aşamazsa, sonuçlar yalın olarak "Anksiyete" veya "Depresyon" olarak hesaplanır.

### 3. Test ve Optimizasyon
![image](https://github.com/user-attachments/assets/679bc18b-a7fd-482f-9841-3a4688e94b02)

Model çıktıları
Test ve değerlendirme tarafında [tranformers-interpret](https://github.com/cdpierse/transformers-interpret) kütüphanesi kullanılmıştır. Bu sayede çıktıdaki kelimelerin hedef etiket ile nasıl çekinlendiği görülmektedir.

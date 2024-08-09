![dilbazlar](https://github.com/user-attachments/assets/5d43fb9f-b59c-4222-ae53-57b89c96ab86)


### Dilbazlar Ekibi Kimdir?
Dilbazlar ekibi, İzmir Bakırçay Üniversitesi Bilgisayar Mühendisliği'nden Ensar Çıtak, Büşra Kurun, Halil İbrahim Hatun ve İzmir Atatürk Lisesi'nden Ahmet Akay'dan oluşmaktadır. 

### Projenin Nedir?
Proje, DSM-5 tabanına bağlı olarak hazırlanmış Anksiyete ve Depresyon hastalıklarından oluşmaktadır. Bu hastalıklar anksiyete, Distimi, PMDD (Prementrüel Disforik Bozukluk), Anksiyete, Agorafobi, Seçici Dilsizlik, 
Panik Bozukluk, Sosyal Anksiyete ve Fobi hastalıklarını kapsamaktadır. 

Bu projede Depresyon ve Anksiyete etiketleri diğer etiketlerden farklı olarak model tespit sonucu değil de eşik değere bağlı olarak seçilir. Mesela Distimi ve PMDD üzerine tahminleme yapan bir depresyon hastalıkları yapay zeka modeli seçilen çıktı eğer konfigüre edilebilir değerin altında kalırsa otomatik Depresyon olarak etiketlenir. 

### Proje aşamaları
![image](https://github.com/user-attachments/assets/164d3075-e0c6-4ec5-96c3-d5ab0d7976aa)

1. Veri Kazıma
2. Model Eğitimi
3. Test ve Optimizasyon
4. Chatbot Seneryosunun oluşturulması


## 1. Veri Kazıma ve Önişleme Aşaması

1.1. Veri 
Veri kazıma aşamasında öncelikle X (Twitter), Reddit sosyal medya platformlarından hashtag ve başlık bağlamındaki verileri etiketli bir şekilde alarak ingilizce bir veri seti elde etmiş olduk. 
Sonraki aşamada HuggingFace'den **Helsinki-NLP/opus-mt-tc-big-en-tr** modelini kullanarak çeviri işleminin ardından veriler türkçeye çevrildi. 

Bunun  yanında, Youtube yorumları ve ekşi sözlük verileri kullanılarak organik türkçe veri elde edilmiştir. Sonra da çevrilmiş veri seti ve organik türkçe veri seti birleştirildi. 

Elde edilen bütünsel veriler LLM model olan Gemini 1.5 flash (free hali kullanılmıştır) ile augmente edildi. 

Bu işlemler sonucunda:
- 89,900 veriye sahip hasta veya normal olarak etiketlenmiş,
- 43,400 veriye sahip anksiyete veya normal olarak etiketlenmiş,
- 57,600 veriye sahip depresyon veya normal olarak etiketlenmiş,
- 27,600 veriye sahip distimi veya PMDD (prementrüal disforik bozukluk) olarak etiketlenmiş,
- 15,700 veriye sahip Agorafobi, Panik, Fobi, Seçici Dilsizlik, Sosyal Anksiyete olarak etiketlenmiş verilere sahip veri setleri Türkçe olarak oluşturulmuş ve Türkiye literatüründe bu tarafta yapılan ilk çalışma gerçekleştirilmiştir.

## 2. Model Eğitimi
Model eğitim sürecinde BERT tabanlı önceden eğitilmiş modeller olan dbmdz/bert-base-turkish-cased ve dbmdz/bert-base-turkish-128k-uncased modelleri kullanılmıştır. 

Sunulan yapıda ilk başta bir içeriğin hasta mı değil mi

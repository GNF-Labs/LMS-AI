### Penjelasan Model
1. Data
Dataset ini memberikan rincian komprehensif tentang keterlibatan, perilaku, aktivitas, dan kinerja siswa terkait kursus online yang telah mereka daftarkan. Setiap catatan dalam koleksi data ini mewakili pengalaman seorang siswa dengan kursus tertentu.

Dataset ini mencakup berbagai metrik dan indikator detail yang menggambarkan interaksi setiap siswa dengan kursus yang mereka pilih. Informasi tersebut termasuk status pendaftaran siswa, apakah mereka melihat konten kursus atau tidak, apakah mereka mendalami materi secara rinci atau hanya melihat sekilas, dan apakah mereka mendapatkan sertifikat setelah menyelesaikan kursus. Link dateaset sebagai berikut : Online Course Student Engagement Metrics (kaggle.com)

2. Model

Model yang digunakan pada LMS yang dikembangkan ini adalah Jodie Model. Jodie adalah model yang menggabungkan embedding statis dan dinamis untuk memprediksi interaksi antara pengguna dan item berdasarkan jaringan interaksi temporal. Ada empat tantangan utama dalam metode pembelajaran embedding dinamis: 
mempertimbangkan sifat statis dan dinamis dari entitas, 
mengurangi kompleksitas waktu dalam memprediksi interaksi, 
mempertahankan ketergantungan temporal, dan 
meningkatkan skalabilitas untuk jutaan interaksi.

JODIE memperbarui embedding pengguna dan item melalui dua RNN yang saling terkait, yang memperhitungkan interaksi sebelumnya antara pengguna dan item. Model ini juga menggunakan operasi proyeksi untuk memprediksi perubahan embedding di masa depan, memanfaatkan lapisan atensi temporal. Selain itu, JODIE menggunakan algoritma t-Batch untuk membuat batch data pelatihan yang independen namun konsisten secara temporal, meningkatkan kecepatan pelatihan.

Hasil eksperimen menunjukkan bahwa JODIE lebih efektif daripada enam algoritma lainnya dalam memprediksi interaksi masa depan dan perubahan keadaan pengguna, dengan peningkatan kinerja hingga 20% untuk prediksi interaksi dan 12% untuk perubahan keadaan pengguna. 

3. Pra Proses Data

JODIE adalah model yang dirancang untuk mempelajari trajektori embedding pengguna dan item seiring waktu. Model ini fokus pada prediksi interaksi pengguna di masa depan berdasarkan urutan interaksi pengguna-item yang terjadi sebelumnya. Pada intinya, setiap pengguna dan item diberikan embedding dinamis yang mencerminkan perubahan preferensi dan karakteristik mereka. Embedding ini diperbarui setelah setiap interaksi, memungkinkan JODIE untuk melacak perubahan temporal.

Dalam model JODIE, terdapat dua jenis embedding: embedding statis dan embedding dinamis. Embedding statis mewakili karakteristik jangka panjang yang stabil, seperti preferensi atau minat tetap dari pengguna. Sementara itu, embedding dinamis menggambarkan sifat yang berubah seiring waktu. Dengan embedding ini, JODIE dapat mengikuti bagaimana pengguna dan item berkembang berdasarkan interaksi yang terus terjadi.

Proses pembaruan embedding ini dilakukan menggunakan dua jaringan saraf berulang, yaitu RNNU untuk pengguna dan RNNI untuk item. Pembaruan embedding pengguna bergantung pada embedding item dari interaksi sebelumnya, dan sebaliknya, menciptakan hubungan timbal balik antara pengguna dan item. Selain itu, salah satu inovasi utama JODIE adalah kemampuan untuk memproyeksikan embedding ke masa depan. Dengan menggunakan embedding saat ini dan waktu yang telah berlalu sejak interaksi terakhir, JODIE memprediksi embedding masa depan pengguna, yang memungkinkan prediksi item yang akan berinteraksi dengan pengguna di waktu mendatang.

Dalam hal pelatihan model, JODIE dirancang untuk meminimalkan perbedaan (L2 loss) antara embedding item yang diprediksi dan embedding item yang sebenarnya. Dengan pendekatan ini, model dapat memprediksi embedding item dengan cepat tanpa perlu menghitung probabilitas interaksi. Selain itu, JODIE menggunakan algoritma batching khusus yang disebut t-Batch. Algoritma ini mempercepat proses pelatihan dengan membagi data ke dalam batch yang tidak saling berbagi pengguna atau item, memungkinkan pemrosesan paralel yang efisien.

Secara keseluruhan, JODIE merupakan model yang mampu memprediksi interaksi pengguna-item dengan akurat, bahkan dalam skenario di mana interaksi bersifat dinamis dan terus berubah seiring waktu. Model ini sangat berguna dalam aplikasi yang memerlukan pemahaman perilaku pengguna yang berkembang, seperti sistem rekomendasi dan prediksi interaksi di jejaring sosial atau platform e-commerce.
 

4. Pelatihan dan Evaluasi
Dalam bagian ini, kami melakukan validasi eksperimental untuk mengukur efektivitas JODIE dalam dua tugas utama: prediksi interaksi course di masa depan dan prediksi perubahan status completion course pengguna. Eksperimen dilakukan pada tiga dataset dan dibandingkan dengan enam model baseline untuk menunjukkan beberapa hal penting. Pertama, JODIE mengungguli baseline dengan margin minimal 20% dalam hal mean reciprocal rank (MRR) dalam memprediksi course berikutnya dan 12% rata-rata dalam memprediksi perubahan status completion course pengguna. Kedua, JODIE terbukti 9,2 kali lebih cepat dibandingkan model DeepCoevolve, serta memiliki kecepatan yang sebanding dengan baseline lainnya. Ketiga, JODIE menunjukkan kinerja yang stabil meskipun data pelatihan dan dimensi embedding yang tersedia terbatas. Terakhir, dalam studi kasus pada dataset course interaction, JODIE mampu memprediksi kemungkinan pengguna (siswa) berhenti belajar lima interaksi sebelum terjadi.

Kami memulai dengan menjelaskan pengaturan eksperimental dan metode baseline yang digunakan, sebelum membahas hasil eksperimen secara rinci. Pengaturan eksperimental dilakukan dengan membagi data berdasarkan urutan waktu untuk mensimulasikan kondisi dunia nyata. Kami melatih semua model pada persentase pertama interaksi (τ%), melakukan validasi pada persentase berikutnya (τv%), dan menguji pada interaksi terakhir yang tersisa. Untuk memastikan perbandingan yang adil, kami menggunakan dimensi embedding dinamis sebesar 128 untuk semua algoritma, dan vektor one-hot untuk embedding statis. Setiap algoritma dilatih selama 50 epoch, dan hasil yang dilaporkan berasal dari kinerja terbaik berdasarkan set validasi.

Kami membandingkan JODIE dengan enam algoritma baseline yang termasuk dalam tiga kategori utama. Pertama, model rekomendasi dengan jaringan saraf berulang (deep recurrent recommender models), yang mencakup RRN, LatentCross, Time-LSTM, dan LSTM standar. Algoritma-algoritma ini adalah yang paling mutakhir dalam sistem rekomendasi dan menghasilkan embedding dinamis untuk pengguna. Kedua, model co-evolusi dinamis (dynamic co-evolution models), di mana JODIE dibandingkan dengan algoritma terkemuka DeepCoevolve, yang telah terbukti lebih baik dari algoritma co-evolusi lainnya. Dalam eksperimen, kami menggunakan 10 sampel negatif per interaksi untuk menjaga efisiensi komputasi. Ketiga, model embedding jaringan temporal (temporal network embedding models), di mana JODIE dibandingkan dengan CTDNE, algoritma mutakhir dalam menghasilkan embedding dari jaringan temporal. Meskipun CTDNE menghasilkan embedding statis, kami melakukan pembaruan embedding setelah setiap interaksi.





### Code setup and Requirements


```
    $ pip install -r requirements.txt
```



### Running the JODIE code

```
   $ python jodie.py --network <network> --model jodie --epochs 50
```

### Evaluate the model

#### Interaction prediction

```
    $ python evaluate_interaction_prediction.py --network <network> --model jodie --epoch 49
```

#### State change prediction


```
   $ python evaluate_state_change_prediction.py --network <network> --model jodie --epoch 49
```

### Run the T-Batch code

```
   $ python tbatch.py --network <network>
```

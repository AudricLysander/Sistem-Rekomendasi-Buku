# Laporan Proyek Machine Learning - Audric Lysander

## Domain Proyek
Buku merupakan salah satu contoh sumber bacaan yang berfungsi sebagai sumber bahan ajaran dalam bentuk materi cetak. [Sumber](https://osf.io/preprints/inarxiv/fmekb/). 

Pada zaman sekarang, buku bukan lagi hanya mengenai pelajaran, namun sudah banyak buku yang dibuat dengan tujuan menghibur para pembaca, seperti genre fiksi, novel, fantasi, misteri, thriller, dan lain sebagainya. Maka dari itu, dibuatlah sistem rekomendasi buku untuk mempermudah pencinta buku dalam memilih buku yang sesuai dengan preferensi mereka.

## Business Understanding

### Problem Statements
- Bagaimana caranya membuat sistem rekomendasi buku berdasarkan genre buku yang serupa?
- Bagaimana caranya membuat sistem rekomendasi buku berdasarkan rating yang telah diberikan oleh pembaca?

### Goals
- Menghasilkan sejumlah rekomendasi buku yang sesuai dengan genre.
- Menghasilkan sejumlah rekomendasi buku sesuai dengan rating yang telah diberikan oleh pembaca.

### Solution Statements
- Sistem rekomendasi akan dibangun dengan teknik *content-based filtering* agar dapat memberikan rekomendasi buku sesuai dengan gener serupa menggunakan *cosine similarity* dan *tf-idf vectorizer* untuk melihat kemiripan dan merepresntasikan fitur penting.
- Sistem rekomendasi akan dibangun dengan teknik *collaborative filtering* agar dapat memberikan rekomendasi buku sesuai dengan rating yang diberikan oleh pembaca menggunakan *RecommenderNet*.

## Data Understanding
Data yang akan digunakan dalam proyek ini adalah dataset goodbooks-10k. Dataset ini merupakan kumpulan data yang berisi peringkat untuk sepuluh ribu buku populer. Umumnya, ada 100 ulasan untuk setiap buku, meskipun beberapa memiliki ulasan yang lebih sedikit. Dataset tersebut didapatkan dari website Kaggle yang dipakai dapat dibuka pada [tautan ini](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k?select=book_tags.csv). 

Pada dataset ini diberikan file:
- book_tags.csv
- books.csv
- ratings.csv
- sample_book.xml
- tag.csv
- to_read.csv

### Variabel-variabel pada Bank Data - Churn Classification:

book_tags.csv
1. goodreads_book_id
2. tag_id
3. count

books.csv
1. id
2. book_id
3. best_book_id
4. work_id
5. books_count
6. isbn
7. isbn13
8. authors
9. original_publication_year
10. original_title
11. title
12. language code
13. average_rating
14. work_ratings_count
15. work_text_reviews_count
16. ratings_1
17. ratings_2
18. ratings_3
19. ratings_4
20. ratings_5
21. image_url
22. small_image_url

ratings.csv
1. book_id
2. user_id
3. rating

sample_book.xml: A sample of raw book XML.

tag.csv
1. tag_id
2. tag_name

to_read.csv
1. user_id
2. book_id

Dilakukan proses *Exploratory Data Analysis* (EDA) untuk dataset books, ratings, book_tags, dan tags.

![](https://github.com/AudricLysander/Sistem-Rekomendasi-Buku/blob/main/asset/EDA.jpg?raw=true)

Dari gambar diatas dapat dilihat total buku, rating, book_tags, dan tags. Data buku yang ada sudah sesuai dengan deskripsi dataset, dimana terdapat 10.000 judul buku, dengan masing-masing judul memiliki kurang lebih 100 ulasan/buku.

## Data Preparation
1. Mengecek Null Values. Pada tahap ini dilakukan pengecekan Null Values pada setiap dataset yang digunakan.

![](https://github.com/AudricLysander/Sistem-Rekomendasi-Buku/blob/main/asset/book_info.jpg?raw=true)

![](https://github.com/AudricLysander/Sistem-Rekomendasi-Buku/blob/main/asset/book_tags_info.jpg?raw=true)

![](https://github.com/AudricLysander/Sistem-Rekomendasi-Buku/blob/main/asset/rating_info.jpg?raw=true)

![](https://github.com/AudricLysander/Sistem-Rekomendasi-Buku/blob/main/asset/tags_info.jpg?raw=true)

Dapat dilihat, bahwa pada dataset books masih terdapat beberapa kolom yang memiliki Null Values. Pada notebook ini tidak akan dilakukan *handling* untuk Null Values, karena kolom tersebut tidak akan dipakai untuk membuat sistem rekomendasi.

Collaborative Filtering

Dilakukan encoding pada data user dan data rating. Selanjutnya hasil encoding akan dijadikan satu DataFrame yang dipakai sebagai data *train* (80%) dan *test* (20%).

## Modeling
Pada tahap ini, akan digunakan Content Based Filtering dan  Collaborative Filtering.

1. Content Based Filtering

Merupakan teknik rekomendasi berdasarkan barang yang populer. Teknik ini dilakukan dengan menggunakan fungsi TfidfVectorizer() untuk mengambil fitur penting dan fitur tersebut akan ditransformasikan dalam matriks. Selanjutnya akan dicari derajat kesamaan menggunakan fungsi cosine_similarity(),  sehingga nanti dapat direkomendasikan kepada user.

Kelebihan dari teknik ini adalah dapat merekomendasikan barang yang belum pernah di ulas, tetapi teknik ini hanya akan memberikan rekomendasi untuk barang yang memiliki kemiripan saja.

Hasil dari sistem rekomendasi yang dibuat menggunakan pendekatan Content Based Filtering adalah sebagai berikut.

Hasil 1
![](https://github.com/AudricLysander/Sistem-Rekomendasi-Buku/blob/main/asset/result1_CBF.jpg?raw=true)

Hasil 2
![](https://github.com/AudricLysander/Sistem-Rekomendasi-Buku/blob/main/asset/result2_CBF.jpg?raw=true)

Dari kedua contoh tersebut, dapat dilihat bahwa sistem akan merekomendasikan buku sesuai dengan persamaan tags yang dimiliki, sehingga tags pada buku yang direkomendasikan akan serupa dengan buku yang kita sukai.

2. Collaborative Filtering

Merupakan teknik rekomendasi berdasarkan prediksi menggunakan deep learning yang mungkin akan disukai oleh pengguna. Teknik ini memerlukan ulasan dari pengguna lain untuk memberikan rekomendasi. Tahap yang perlu dilakukan sebelum menggunakan teknik ini adalah tahap preparation untuk encoding dataset, melakukan split data train dan test, dan membuat class ReccomenderNet untuk memberikan hasil rekomendasi.

Kelebihan dari teknik ini adalah dapat merekomendasikan barang dalam kondisi sulitnya melakukan analisis, namun pendekatan ini memerlukan data ulasan untuk memberikan rekomendasi.

Hasil dari sistem rekomendasi yang dibuat menggunakan pendekatan Collaborative Filtering adalah sebagai berikut.

![](https://github.com/AudricLysander/Sistem-Rekomendasi-Buku/blob/main/asset/result_CF.jpg?raw=true)

## Evaluation

1. Content Based Filtering

Evaluasi pada teknik ini menggunakan metrik presisi yang dihitung berdasarkan rekomendasi yang memiliki tag yang serupa. Nilai metrik presisi dapat dihitung dengan rumus sebagai berikut.

![](https://hasty.ai/media/pages/docs/mp-wiki/metrics/accuracy/fcbf093d04-1653642321/11.png)

- *Accuracy*: akurasi
- *Number of correct predictions*: Jumlah data benar
- *Total number of predictions*: Jumlah data prediksi


2. Collaborative Filtering

Evaluasi pada teknik ini menggunakan *Root Mean Squared Error* (RMSE). Nilai RMSE dapat dihitung dengan rumus sebagai berikut.

![](https://community.qlik.com/legacyfs/online/128958_2016-06-23%2013_45_36-Root%20Mean%20Squared%20Error%20_%20Kaggle.png)

- N: Jumlah data observasi
- $y_{i}$: Nilai prediksi
- $\hat{y}_{i}$: Nilai aktual

## Kesimpulan
Sistem rekomendasi yang menggunakan teknik Content Based Filtering dan Collaborative Filtering sama-sama dapat memberikan pengguna rekomendasi suatu barang. Penggunaan kedua teknik tersebut memiliki kelebihan dan kekurangan masing-masing, sehingga dapat kita pakai sesuai dengan kebutuhan dan data yang tersedia untuk mendapatkan hasil rekomendasi yang maksimal.

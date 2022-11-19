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

![](data:image/png;base64,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)

- N: Jumlah data observasi
- $$\hat{y}_{i}$$: Nilai prediksi
- $$y_{i}$$: Nilai aktual

## Kesimpulan
Sistem rekomendasi yang menggunakan teknik Content Based Filtering dan Collaborative Filtering sama-sama dapat memberikan pengguna rekomendasi suatu barang. Penggunaan kedua teknik tersebut memiliki kelebihan dan kekurangan masing-masing, sehingga dapat kita pakai sesuai dengan kebutuhan dan data yang tersedia untuk mendapatkan hasil rekomendasi yang maksimal.

# Laporan Proyek Machine Learning - Audric Lysander

## Domain Proyek
Buku merupakan salah satu contoh sumber bacaan yang berfungsi sebagai sumber bahan ajaran dalam bentuk materi cetak [[1](https://osf.io/preprints/inarxiv/fmekb/)]. Pada zaman sekarang, buku bukan lagi hanya mengenai pelajaran, namun sudah banyak buku yang dibuat dengan tujuan menghibur para pembaca, seperti genre fiksi, novel, fantasi, misteri, thriller, dan lain sebagainya. Maka dari itu, dibuatlah sistem rekomendasi buku untuk mempermudah pencinta buku dalam memilih buku yang sesuai dengan preferensi mereka.

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
1. goodreads_book_id: id buku
2. tag_id: id tag (genre) buku
3. count: letak buku

books.csv
1. id: id data
2. book_id: id buku
3. best_book_id: id buku
4. work_id: id penulis
5. books_count: total buku yang tersedia
6. isbn: *International Standard Book Number*
7. isbn13: *International Standard Book Number* dengan 13 digit angka
8. authors: nama penulai
9. original_publication_year: tahun terbit
10. original_title: judul orisinil
11. title: judul buku
12. language_code: kode bahasa
13. average_rating: rata-rata ulasan
14. work_ratings_count: total ulasan
15. work_text_reviews_count: total ulasan dalam teks
16. ratings_1: total ulasan bintang 1
17. ratings_2: total ulasan bintang 2
18. ratings_3: total ulasan bintang 3
19. ratings_4: total ulasan bintang 4
20. ratings_5: total ulasan bintang 5
21. image_url: link cover buku
22. small_image_url: link cover buku versi kecil

ratings.csv
1. book_id: id buku
2. user_id: id user
3. rating: ulasan user

sample_book.xml: A sample of raw book XML.

tag.csv
1. tag_id: id tag (genre)
2. tag_name: nama tag (genre)

to_read.csv
1. user_id: id user
2. book_id: id buku

Dilakukan proses *Exploratory Data Analysis* (EDA) untuk dataset books, ratings, book_tags, dan tags.

| Dataset   | Jumlah data |
|-----------|-------------|
| Book      | 10000       |
| Rating    | 981756      |
| Book Tags | 34252       |
| Tags      | 34252       |

Dari gambar diatas dapat dilihat total buku, rating, book_tags, dan tags. Data buku yang ada sudah sesuai dengan deskripsi dataset, dimana terdapat 10.000 judul buku, dengan masing-masing judul memiliki kurang lebih 100 ulasan/buku.

## Data Preparation
1. Mengecek Null Values. Pada tahap ini dilakukan pengecekan Null Values pada setiap dataset yang digunakan.

Books
| Fitur                     | Null Values |
|---------------------------|-------------|
| id                        | 0           |
| book_id                   | 0           |
| best_book_id              | 0           |
| work_id                   | 0           |
| books_count               | 0           |
| isbn                      | 700         |
| isbn13                    | 585         |
| authors                   | 0           |
| original_publication_year | 21          |
| original_title            | 585         |
| title                     | 0           |
| language_code             | 1084        |
| average_rating            | 0           |
| ratings_count             | 0           |
| work_ratings_count     | 0           |
| work_text_review_count | 0           |
| ratings_1              | 0           |
| ratings_2              | 0           |
| ratings_3              | 0           |
| ratings_4              | 0           |
| ratings_5              | 0           |
| image_url              | 0           |
| small_image_url        | 0           |

Book Tags
| Fitur             | Null Values |
|-------------------|-------------|
| goodreads_book_id | 0           |
| tag_id            | 0           |
| count             | 0           |

Rating
| Fitur       | Null Values |
|-------------|-------------|
| book_id     | 0           |
| user_id     | 0           |
| countrating | 0           |

Tags
| Fitur    | Null Values |
|----------|-------------|
| tag_id   | 0           |
| tag_name | 0           |

Dapat dilihat, bahwa pada dataset books masih terdapat beberapa kolom yang memiliki Null Values. Pada notebook ini tidak akan dilakukan *handling* untuk Null Values, karena kolom tersebut tidak akan dipakai untuk membuat sistem rekomendasi.

Collaborative Filtering

Dilakukan encoding pada data user dan data rating. Selanjutnya hasil encoding akan dijadikan satu DataFrame yang dipakai sebagai data *train* (80%) dan *test* (20%).

## Modeling
Pada tahap ini, akan digunakan Content Based Filtering dan  Collaborative Filtering.

1. Content Based Filtering

Merupakan teknik rekomendasi berdasarkan barang yang populer. Teknik ini dilakukan dengan menggunakan fungsi TfidfVectorizer() untuk mengambil fitur penting seperti title serta tags, lalu fitur tersebut akan ditransformasikan dalam matriks. Pada tahap ini, kita akan menginisialisasi TfidfVectorizer terlebih dahulu, dilanjutkan dengan melakukan perhitungan idf pada data tag_name dan menjadikannya kedalam array, sehingga setiap nilai unik pada tag_name akan menjadi 1 array dan ditransformasikan ke dalam bentuk matriks. Untuk menghasilkan vektor tf-idf dalam bentuk matriks bisa menggunakan fungsi todense(). Matriks tersebut akan menunjukan korelasi antara tag_name dengan title. Selanjutnya akan dicari derajat kesamaan menggunakan fungsi cosine_similarity(),  sehingga nanti dapat direkomendasikan kepada user. Pada tahap ini, akan dihitung cosine similarity dataframe yang diperoleh pada tahapan sebelumnya. Selanjutnya matriks kesamaan setiap title dengan menampilkan title buku dalam 5 sampel kolom dan 10 sampel baris. Dengan cosine, kita bisa mengidentifikasi kesamaan anatara satu buku dengan buku lainnya.

Kelebihan dari teknik ini adalah dapat merekomendasikan barang yang belum pernah di ulas, tetapi teknik ini hanya akan memberikan rekomendasi untuk barang yang memiliki kemiripan saja.

Hasil dari sistem rekomendasi yang dibuat menggunakan pendekatan Content Based Filtering adalah sebagai berikut.

Hasil 1
Pencarian Buku
| title                                   | tag_name                                          |
|-----------------------------------------|---------------------------------------------------|
| Flowers in the Attic (Dollanganger, #1) | read-in-2014,family,dnf,library,currently-read... |

Hasil Rekomendasi
| title                                   | tag_name                                          |
|-----------------------------------------|---------------------------------------------------|
| Petals on the Wind (Dollanganger, #2)   | read-in-2014,family,dnf,library,my-bookshelf,f... |
| If There Be Thorns (Dollanganger, #3)   | vcandrews,read-in-2014,family,library,my-books... |
| Seeds of Yesterday (Dollanganger, #4)   | vcandrews,read-in-2014,family,own-it,library,m... |
| Garden of Shadows (Dollanganger, #5)    | vcandrews,read-in-2014,read-as-a-kid,family,se... |
| My Sweet Audrina (Audrina, #1)          | vcandrews,family,dnf,stand-alones,own-it,libra... |

Hasil 2
Pencarian Buku
| title                                          | tag_name                                          |
|------------------------------------------------|---------------------------------------------------|
| Soulless (Parasol Protectorate, #1)            | read-in-2014,dnf,funny,sff,london,library,para... |

Hasil Rekomendasi
| title                                          | tag_name                                          |
|------------------------------------------------|---------------------------------------------------|
| Changeless (Parasol Protectorate, #2)          | read-in-2014,scotland,funny,sff,paranormal-fan... |
| Blameless (Parasol Protectorate, #3)           | read-in-2014,funny,sff,london,library,paranorm... |
| Timeless (Parasol Protectorate, #5)            | read-in-2014,own-it,funny,sff,london,library,p... |
| Curtsies & Conspiracies (Finishing School, #2) | read-in-2014,spy,funny,library,audio-books,fem... |
| Heartless (Parasol Protectorate, #4)           | read-in-2014,2011-reads,own-it,funny,sff,londo... |

Dari kedua contoh tersebut, dapat dilihat bahwa sistem akan merekomendasikan buku sesuai dengan persamaan tags yang dimiliki, sehingga tags pada buku yang direkomendasikan akan serupa dengan buku yang kita sukai.

2. Collaborative Filtering

Merupakan teknik rekomendasi berdasarkan prediksi menggunakan deep learning yang mungkin akan disukai oleh pengguna. Teknik ini memerlukan ulasan dari pengguna lain untuk memberikan rekomendasi. Tahap yang perlu dilakukan sebelum menggunakan teknik ini adalah tahap preparation untuk encoding dataset, melakukan split data train dan test, dan membuat class RecommenderNet untuk memberikan hasil rekomendasi. 

Proses sebelum encoding dilakukan adalah mengubah user_id menjadi list tanpa nilai yang sama, lalu dilakukan encoding angka ke user_id, begitupula dengan fitur book_id. Lalu dilakukan split data train (80%) dan test (20%). Selanjutnya buat rating dalam skala 0 sampai 1 agar mudah untuk dilakukan proses training. Proses berikutnya akan dibuat class RecommenderNet dengan keras Model class. Di dalam RecommenderNet ini akan dihitung skor kecocokan antara pengguna dan buku dengan teknik embedding. Pertama akan dilakukan proses embedding pada user dan buku. Selanjutnya akan dilakukan operasi perkalian dot produk antara embedding user dan buku. Selain itu juga dapat menambahkan bias untuk setiap user dan buku. Skor kecocokan ditetapkan dalam skala [0, 1] dengan fungsi aktivasi sigmoid.

Kelebihan dari teknik ini adalah dapat merekomendasikan barang dalam kondisi sulitnya melakukan analisis, namun pendekatan ini memerlukan data ulasan untuk memberikan rekomendasi.

Hasil dari sistem rekomendasi yang dibuat menggunakan pendekatan Collaborative Filtering adalah sebagai berikut.

Rekomendasi untuk user: 25214

| title                                                                       | tag_name                                                                   |
|-----------------------------------------------------------------------------|----------------------------------------------------------------------------|
| Harry Potter and the Order of the Phoenix (Harry Potter, #5)                | read-in-2014,own-it,children-s-literature,harry-potter-series              |
| The Hitchhiker's Guide to the Galaxy (Hitchhiker's Guide to the Galaxy, #1) | read-in-2014,dnf,own-it,funny,library,audio-books                          |
| The Ultimate Hitchhiker's Guide to the Galaxy                               | own-it,funny,sff,library,my-bookshelf,currently-reading,i-own,sf           |
| Neither Here nor There: Travels in Europe                                   | read-in-2014,mystery-crime-thriller,library,california,currently-reading   |
| Darwin's Dangerous Idea: Evolution and the Meanings of Life                 | books-to-buy,essay,own-it,ideas,library,filosofie                          |
| Built to Last: Successful Habits of Visionary Companies                     | business-marketing,professional-development,own-it                         |
| Beloved                                                                     | women-writers,literary,for-school,own-it,rory-gilmore-reading-list         |
| Cry, the Beloved Country                                                    | african,read-in-2014,literary,for-school,own-it                            |
| Survival in Auschwitz                                                       | world-war-two,own-it,library,italian-authors,currently-reading,biographies |

## Evaluation

1. Content Based Filtering

Evaluasi pada teknik ini menggunakan metrik presisi yang dihitung berdasarkan rekomendasi yang memiliki tag yang serupa. Nilai metrik presisi dapat dihitung dengan rumus sebagai berikut.

![](https://hasty.ai/media/pages/docs/mp-wiki/metrics/accuracy/fcbf093d04-1653642321/11.png)

- *Accuracy*: akurasi
- *Number of correct predictions*: Jumlah data benar
- *Total number of predictions*: Jumlah data prediksi

Dari kedua hasil rekomendasi yang didapatkan, dapat diketahui bahwa sistem rekomendasi memberikan judul buku dengan tags yang serupa pada kedua percobaan diatas, sehingga dapat disimpulkan bahwa sistem memiliki nilai 100% pada nilai akurasi.

2. Collaborative Filtering

Evaluasi pada teknik ini menggunakan *Root Mean Squared Error* (RMSE). Nilai RMSE dapat dihitung dengan rumus sebagai berikut.

![](https://community.qlik.com/legacyfs/online/128958_2016-06-23%2013_45_36-Root%20Mean%20Squared%20Error%20_%20Kaggle.png)

- N: Jumlah data observasi
- $y_{i}$: Nilai prediksi
- $\hat{y}_{i}$: Nilai aktual

![](https://github.com/AudricLysander/Sistem-Rekomendasi-Buku/blob/main/asset/evaluasi.jpg?raw=True)

Didapatkan nilai RMSE yang semakin menurun pada tiap epoch-nya, pada epoch terakhir (ke-20), didapatkan error pada nilai train sebesar 0,245 dan test sebesar 0,246. Dari grafik plot tersebut, dapat dikatakan sebagai goodfit karena nilai train dan test tidak berbeda jauh, dimana hanya berbeda 0,001 saja, sehingga dapat dikatakan goodfit.

## Kesimpulan
Sistem rekomendasi yang menggunakan teknik Content Based Filtering dan Collaborative Filtering sama-sama dapat memberikan pengguna rekomendasi suatu barang. Penggunaan kedua teknik tersebut memiliki kelebihan dan kekurangan masing-masing, sehingga dapat kita pakai sesuai dengan kebutuhan dan data yang tersedia untuk mendapatkan hasil rekomendasi yang maksimal.

## Referensi
[1] Y. Yanti, A. Asrizal, and Festiyed, “Pengertian, jenis-jenis, Dan Karakteristik Bahan ajar Cetak Meliputi hand out, Modul, buku (Diktat, Buku Ajar, Buku Teks), LKS Dan Pamflet,” 2019. 

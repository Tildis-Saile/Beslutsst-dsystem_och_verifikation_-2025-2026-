# Beslutsst-dsystem_och_verifikation_-2025-2026-Kursprojekt

## 1. EDA & Rekommendationssystem

Vi använder oss av [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset/code) för att laga ett rekommendationssystem baserat på Spotify.

Vi har lagat två separata .py filer för att rekommendera på Content-based och med Collaborative filtering.


<details>
<summary>
Collaborative filtering output
</summary>

```bash
Initializing collaborative filtering recommender...
Creating user-item matrix from implicit feedback...
User-item matrix created: (114, 89741)
Number of users: 114
Number of items: 89741
Matrix density: 0.0111

Fitting models...
Fitting SVD model with 20 components...
SVD explained variance ratio: 0.3225
Fitting NMF model with 20 components...
NMF reconstruction error: 132.4452
Fitting user-based CF with 10 neighbors...
Fitting item-based CF with 10 neighbors...

Testing recommendations for 5 users...

--- Recommendations for genre_acoustic ---
SVD: ['孤勇者 - 《英雄聯盟:雙城之戰》動畫劇集中文主題曲', '獨家記憶', '這世界那麼多人 - 電影《我要我們在一起》主題曲']
User-based: ['Piano Man', 'drivers license - Piano Arrangement', 'Vienna']
Item-based: ['AT アイリッド', 'Please Come Home For Christmas - Album Version / Addtl. Strings', 'Back Door Santa']

--- Recommendations for genre_afrobeat ---
SVD: ['Look at the Sky', 'Dead To Me - Slow + Reverb', 'Goodbye To A World']
User-based: ['Dead To Me - Slow + Reverb', 'Goodbye To A World', 'All That Really Matters']
Item-based: ['Marcus the Prophet', 'Os Alquimistas Estão Chegando Os Alquimistas', 'Alcohol']

--- Recommendations for genre_alt-rock ---
SVD: ['Take A Look Around', 'Behind Blue Eyes', 'Hail to the King']
User-based: ['In the End', 'Take A Look Around', 'Behind Blue Eyes']
Item-based: ['Banderitas y Globos', 'Debede', 'Angel Of Death']

--- Recommendations for genre_alternative ---
SVD: ['Deutschland', 'Like a Stone', 'Drive']
User-based: ['Like a Stone', 'Man in the Box', 'Drive']
Item-based: ['Teardrinker', 'Teardrinker', 'Walk']

--- Recommendations for genre_ambient ---
SVD: ['Our God - New Recording', 'Way Maker - Live', '10,000 Reasons (Bless The Lord) - Live']
User-based: ['Losing My Religion', 'Something Bout That Feeling', 'Heat Waves']
Item-based: ["Don't Dream It's Over", 'Virginia (Wind in the Night)', "Don't Dream It's Over"]

Evaluating models...
Evaluating collaborative filtering model...
   Processing preferences for up to 20 users...
   Processing user 1/20...
   Processing user 11/20...
   Found preferences for 20 users
Evaluating svd...
Evaluating user_based...
Evaluating item_based...

Evaluation Results:
svd: {'accuracy': 1.0, 'total_recommendations': 100, 'correct_matches': 100}
user_based: {'accuracy': 1.0, 'total_recommendations': 100, 'correct_matches': 100}
item_based: {'accuracy': 1.0, 'total_recommendations': 95, 'correct_matches': 95}

Collaborative filtering demo completed successfully!
```

</details>

<details>
<summary>
Content based output
</summary>

```bash
E:\Repo\Beslutsst-dsystem_och_verifikation_-2025-2026-\scripts>python basic_recommender.py
Recommendations for 'Back In Black':
- Thunderstruck
- Highway to Hell
- T.N.T.
- You Shook Me All Night Long
- Shoot to Thrill
```

</details>

<details>
<summary>
Hybrid output
</summary>

```bash
E:\Repo\Beslutsst-dsystem_och_verifikation_-2025-2026-\scripts>python hybrid_recommender.py
Creating user-item matrix from implicit feedback...
User-item matrix created: (114, 89741)
Number of users: 114, Items: 89741
Matrix density: 0.0111
Fitting collaborative filtering models...
Fitting SVD model with 20 components...
SVD explained variance ratio: 0.3225
Fitting user-based CF with 10 neighbors...
Fitting item-based CF with 10 neighbors...
Using collaborative filtering user: 'genre_hard-rock' for track 'Back In Black'
SVD recommendations: ['Zombie (Acoustic)', 'Can You Feel My Heart', 'No One Knows', 'The Rumbling', 'Your Betrayal']
User-based recommendations: ['Zombie (Acoustic)', 'No One Knows', 'The Diary Of Jane', 'No Rain', 'Californication']
Item-based recommendations: ['Still Worth Fighting For', "Don't Stay", 'Courtesy Call', 'Feel Invincible', 'One Last Breath']
Hybrid Recommendations for 'Back In Black':
- Feel Invincible
- You Shook Me All Night Long
- One Last Breath
- Your Betrayal
- T.N.T.
```

</details>


<details>
<summary>
EDA Output
</summary>

```bash
E:\Repo\Beslutsst-dsystem_och_verifikation_-2025-2026-\scripts>python eda.py
Loading Spotify tracks dataset from: E:\Repo\Beslutsst-dsystem_och_verifikation_-2025-2026-\data\dataset.csv
Dataset loaded successfully!
Dataset shape: (114000, 21)
Memory usage: 54.31 MB
 Starting Comprehensive Spotify Dataset EDA Analysis
================================================================================

============================================================
 BASIC DATASET INFORMATION
============================================================

 Dataset Dimensions: 114,000 rows × 21 columns

 Column Names:
   1. Unnamed: 0
   2. track_id
   3. artists
   4. album_name
   5. track_name
   6. popularity
   7. duration_ms
   8. explicit
   9. danceability
  10. energy
  11. key
  12. loudness
  13. mode
  14. speechiness
  15. acousticness
  16. instrumentalness
  17. liveness
  18. valence
  19. tempo
  20. time_signature
  21. track_genre

 Data Types:
Unnamed: 0            int64
track_id             object
artists              object
album_name           object
track_name           object
popularity            int64
duration_ms           int64
explicit               bool
danceability        float64
energy              float64
key                   int64
loudness            float64
mode                  int64
speechiness         float64
acousticness        float64
instrumentalness    float64
liveness            float64
valence             float64
tempo               float64
time_signature        int64
track_genre          object
dtype: object

 Missing Values:
            Missing Count  Missing %
artists                 1   0.000877
album_name              1   0.000877
track_name              1   0.000877

 Basic Statistics for Numeric Columns:
          Unnamed: 0     popularity   duration_ms  ...        valence          tempo  time_signature
count  114000.000000  114000.000000  1.140000e+05  ...  114000.000000  114000.000000   114000.000000
mean    56999.500000      33.238535  2.280292e+05  ...       0.474068     122.147837        3.904035
std     32909.109681      22.305078  1.072977e+05  ...       0.259261      29.978197        0.432621
min         0.000000       0.000000  0.000000e+00  ...       0.000000       0.000000        0.000000
25%     28499.750000      17.000000  1.740660e+05  ...       0.260000      99.218750        4.000000
50%     56999.500000      35.000000  2.129060e+05  ...       0.464000     122.017000        4.000000
75%     85499.250000      50.000000  2.615060e+05  ...       0.683000     140.071000        4.000000
max    113999.000000     100.000000  5.237295e+06  ...       0.995000     243.372000        5.000000

[8 rows x 15 columns]

============================================================
 TRACK ANALYSIS
============================================================

 Total unique tracks: 73,608
 Duplicate tracks: 40,391

 Top 10 Most Popular Tracks:
   1. Unholy (feat. Kim Petras) - Sam Smith;Kim Petras (100)
   2. Unholy (feat. Kim Petras) - Sam Smith;Kim Petras (100)
   3. Quevedo: Bzrp Music Sessions, Vol. 52 - Bizarrap;Quevedo (99)
   4. I'm Good (Blue) - David Guetta;Bebe Rexha (98)
   5. I'm Good (Blue) - David Guetta;Bebe Rexha (98)
   6. La Bachata - Manuel Turizo (98)
   7. La Bachata - Manuel Turizo (98)
   8. I'm Good (Blue) - David Guetta;Bebe Rexha (98)
   9. La Bachata - Manuel Turizo (98)
  10. La Bachata - Manuel Turizo (98)

 Track Duration Analysis:
  Average duration: 3.80 minutes
  Median duration: 3.55 minutes
  Shortest track: 0.00 minutes
  Longest track: 87.29 minutes

 Explicit content: 9,747 tracks (8.6%)

============================================================
 GENRE ANALYSIS
============================================================

 Total unique genres: 114

 Top 15 Genres by Track Count:
   1. acoustic              1,000 tracks (  0.9%)
   2. punk-rock             1,000 tracks (  0.9%)
   3. progressive-house     1,000 tracks (  0.9%)
   4. power-pop             1,000 tracks (  0.9%)
   5. pop                   1,000 tracks (  0.9%)
   6. pop-film              1,000 tracks (  0.9%)
   7. piano                 1,000 tracks (  0.9%)
   8. party                 1,000 tracks (  0.9%)
   9. pagode                1,000 tracks (  0.9%)
  10. opera                 1,000 tracks (  0.9%)
  11. new-age               1,000 tracks (  0.9%)
  12. mpb                   1,000 tracks (  0.9%)
  13. minimal-techno        1,000 tracks (  0.9%)
  14. metalcore             1,000 tracks (  0.9%)
  15. metal                 1,000 tracks (  0.9%)

 Most Popular Genres (by average popularity):
   1. pop-film             Avg:  59.3 (1000.0 tracks)
   2. k-pop                Avg:  56.9 (1000.0 tracks)
   3. chill                Avg:  53.6 (1000.0 tracks)
   4. sad                  Avg:  52.4 (1000.0 tracks)
   5. grunge               Avg:  49.6 (1000.0 tracks)
   6. indian               Avg:  49.5 (1000.0 tracks)
   7. anime                Avg:  48.8 (1000.0 tracks)
   8. emo                  Avg:  48.1 (1000.0 tracks)
   9. sertanejo            Avg:  47.9 (1000.0 tracks)
  10. pop                  Avg:  47.6 (1000.0 tracks)

============================================================
 ARTIST ANALYSIS
============================================================

 Total unique artists: 31,437

 Top 15 Most Prolific Artists (by track count):
   1. The Beatles                     279 tracks
   2. George Jones                    271 tracks
   3. Stevie Wonder                   236 tracks
   4. Linkin Park                     224 tracks
   5. Ella Fitzgerald                 222 tracks
   6. Prateek Kuhad                   217 tracks
   7. Feid                            202 tracks
   8. Chuck Berry                     190 tracks
   9. Håkan Hellström                 183 tracks
  10. OneRepublic                     181 tracks
  11. The Beach Boys                  176 tracks
  12. my little airport               171 tracks
  13. Elvis Presley                   169 tracks
  14. Charlie Brown Jr.               169 tracks
  15. Red Hot Chili Peppers           159 tracks

 Most Popular Artists (by average popularity):
   1. Bad Bunny                      Avg:  87.1 (48.0 tracks)
   2. Marshmello;Khalid              Avg:  84.0 (10.0 tracks)
   3. Måneskin                       Avg:  83.7 (12.0 tracks)
   4. Ariana Grande                  Avg:  81.5 (11.0 tracks)
   5. Clairo                         Avg:  77.0 (10.0 tracks)
   6. Farruko                        Avg:  76.7 (13.0 tracks)
   7. Radiohead                      Avg:  76.7 (20.0 tracks)
   8. Travis Scott                   Avg:  76.3 (18.0 tracks)
   9. Labrinth                       Avg:  76.1 (12.0 tracks)
  10. The Neighbourhood              Avg:  75.9 (56.0 tracks)

============================================================
 AUDIO FEATURES ANALYSIS
============================================================

 Audio Features Summary Statistics:
       danceability      energy         key    loudness  ...  instrumentalness    liveness     valence       tempo
count    114000.000  114000.000  114000.000  114000.000  ...        114000.000  114000.000  114000.000  114000.000
mean          0.567       0.641       5.309      -8.259  ...             0.156       0.214       0.474     122.148
std           0.174       0.252       3.560       5.029  ...             0.310       0.190       0.259      29.978
min           0.000       0.000       0.000     -49.531  ...             0.000       0.000       0.000       0.000
25%           0.456       0.472       2.000     -10.013  ...             0.000       0.098       0.260      99.219
50%           0.580       0.685       5.000      -7.004  ...             0.000       0.132       0.464     122.017
75%           0.695       0.854       8.000      -5.003  ...             0.049       0.273       0.683     140.071
max           0.985       1.000      11.000       4.532  ...             1.000       1.000       0.995     243.372

[8 rows x 11 columns]

 Audio Features Correlation Matrix:

 Top 10 Strongest Feature Correlations:
   1. energy          <-> loudness         0.762
   2. energy          <-> acousticness    -0.734
   3. loudness        <-> acousticness    -0.590
   4. danceability    <-> valence          0.477
   5. loudness        <-> instrumentalness -0.433
   6. instrumentalness <-> valence         -0.324
   7. loudness        <-> valence          0.280
   8. danceability    <-> loudness         0.259
   9. energy          <-> valence          0.259
  10. energy          <-> tempo            0.248

============================================================
 CREATING VISUALIZATIONS
============================================================
 Visualizations saved as 'spotify_eda_visualizations.png' in the resources folder.

============================================================
 EDA SUMMARY REPORT
============================================================

 DATA QUALITY SUMMARY:
  • Total tracks: 114,000
  • Total features: 21
  • Data completeness: 100.0%
  • Missing values: 3

 GENRE INSIGHTS:
  • Unique genres: 114
  • Most common genre: acoustic (1,000 tracks)
  • Genre diversity: 0.9% of tracks are acoustic

 AUDIO FEATURES INSIGHTS:
  • High energy tracks (>0.8): 37,880 (33.2%)
  • High danceability tracks (>0.8): 8,768 (7.7%)
  • Acoustic tracks (>0.8): 15,792 (13.9%)

 POPULARITY INSIGHTS:
  • Average popularity: 33.2
  • High popularity tracks (>70): 4,846 (4.3%)

 DURATION INSIGHTS:
  • Average duration: 3.8 minutes
  • Short tracks (<2 min): 6,253 (5.5%)
  • Long tracks (>5 min): 16,348 (14.3%)

 EDA analysis completed successfully!

 EDA Analysis Complete!
 Check 'spotify_eda_visualizations.png' for visualizations
```

</details>


Använd valfritt data set, utforska och analysera ert data set (EDA).

Bygg ett rekommendationssystem baserat på innehållsbaserad filtrering (Content-based filtering), högre komplexitet ger högre poäng.

Utöka eller gör ett till rekommendationssystemet med kollaborativ filtrering (Collaborative filtering).

Gör en rapport. Kan vara en .md fil med skärmdumpar eller .py/.ipynb om man har mycket grafer.
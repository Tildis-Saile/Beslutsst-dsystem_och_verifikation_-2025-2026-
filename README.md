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

</details>


Använd valfritt data set, utforska och analysera ert data set (EDA).

Bygg ett rekommendationssystem baserat på innehållsbaserad filtrering (Content-based filtering), högre komplexitet ger högre poäng.

Utöka eller gör ett till rekommendationssystemet med kollaborativ filtrering (Collaborative filtering).

Gör en rapport. Kan vara en .md fil med skärmdumpar eller .py/.ipynb om man har mycket grafer.
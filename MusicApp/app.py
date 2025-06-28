from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

try:
    songs_df = pd.read_csv('./my_songs.csv')
    processed_data = np.load('./processed_song_features.npy')
except FileNotFoundError as e:
    print(f"File not found: {e}")
    exit(1)

knn_model = NearestNeighbors(n_neighbors=9, metric='euclidean')
knn_model.fit(processed_data)

def recommend_next_song(song_id):
    track = songs_df[songs_df['track_id'] == song_id]

    if track.empty:
        return []

    song_index = track.index[0]
    distances, indices = knn_model.kneighbors(processed_data[song_index].reshape(1, -1))

    recommended_songs = songs_df.iloc[indices[0][1:]]
    return recommended_songs.sort_values(by='popularity', ascending=False, na_position='last').to_dict(orient="records")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        search_query = request.form['search_query'].strip()
        if search_query:
            filtered_df = songs_df[songs_df["track_name"].str.contains(search_query, case=False, na=False)]
            return render_template("index.html", songs=filtered_df.to_dict(orient="records"), query=search_query)
    return render_template("index.html", songs=[])

@app.route('/recommend/<song_id>')
def recommend(song_id):
    selected_song = songs_df[songs_df["track_id"] == song_id].iloc[0].to_dict()
    recommendations = recommend_next_song(song_id)
    return render_template("recommendations.html", song=selected_song, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)

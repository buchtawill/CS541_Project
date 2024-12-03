import csv
import os
import re
from collections import Counter

# File paths
genres_csv = "fma_metadata/genres.csv"
tracks_csv = "fma_metadata/tracks_trimmed.csv"
mp3_folder = "fma_medium"
output_file = "mp3_titles_and_genres_medium.txt"

def load_genres(genres_csv):
    """Load genres from genres.csv into a dictionary, accounting for top-level genres."""
    genres = {}
    genre_track_count = {}
    top_level_map = {}
    
    with open(genres_csv, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                genre_id = int(row['genre_id'])
                title = row['title']
                top_level = int(row['top_level'])
                num_tracks = int(row['#tracks'])
                
                # Map genre_id to title
                genres[genre_id] = title
                # Track number of tracks per genre
                genre_track_count[genre_id] = num_tracks
                # Map genre_id to its top-level genre ID
                top_level_map[genre_id] = top_level
            except ValueError:
                continue  # Skip invalid rows
                
    return genres, genre_track_count, top_level_map

def load_tracks(tracks_csv, genre_track_count, top_level_map):
    """Load track genre mapping from tracks_trimmed.csv, ensuring top-level genres are used."""
    tracks = {}
    
    with open(tracks_csv, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                track_id = int(row['track_id'])
                genre_ids = re.findall(r'\d+', row['genre_ids'])
                if genre_ids:
                    # Select the genre ID with the most tracks
                    genre_id = max((int(gid) for gid in genre_ids), key=lambda gid: genre_track_count.get(gid, 0))
                    # Map to top-level genre
                    top_level_id = top_level_map.get(genre_id, genre_id)
                    tracks[track_id] = top_level_id
            except (ValueError, KeyError):
                continue  # Skip invalid rows
                
    return tracks

def match_mp3_files(mp3_folder, tracks, genres):
    """Match MP3 files with their genres, accounting for nested folder structure."""
    mp3_data = []
    for folder in os.listdir(mp3_folder):
        folder_path = os.path.join(mp3_folder, folder)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith('.mp3'):
                    try:
                        track_id = int(os.path.splitext(filename)[0])
                        genre_id = tracks.get(track_id)
                        if genre_id is not None:
                            genre_title = genres.get(genre_id, "Unknown Genre")
                            mp3_data.append((os.path.join(folder, filename), genre_id, genre_title))
                    except ValueError:
                        continue  # Skip any invalid filenames
    return mp3_data

def count_genres(mp3_folder, tracks, genres):
    """Count the occurrences of each genre in the MP3 folder."""
    genre_counts = Counter()
    for folder in os.listdir(mp3_folder):
        folder_path = os.path.join(mp3_folder, folder)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith('.mp3'):
                    try:
                        track_id = int(os.path.splitext(filename)[0])
                        genre_id = tracks.get(track_id)
                        if genre_id is not None:
                            genre_title = genres.get(genre_id, "Unknown Genre")
                            genre_counts[genre_title] += 1
                    except ValueError:
                        continue  # Skip invalid filenames
    return genre_counts

def write_output(output_file, mp3_data):
    """Write the MP3 file titles and their genres to an output file."""
    with open(output_file, 'w') as file:
        for mp3_title, genre_id, genre in mp3_data:
            # file_name = os.path.basename(mp3_title)
            file.write(f"{mp3_title}, {genre_id}, {genre}\n")
            
def main():
    genres, genre_track_count, top_level_map = load_genres(genres_csv)
    tracks = load_tracks(tracks_csv, genre_track_count, top_level_map)
    genre_counts = count_genres(mp3_folder, tracks, genres)
    
    mp3_data = match_mp3_files(mp3_folder, tracks, genres)
    write_output(output_file, mp3_data)
    print(f"Output written to {output_file}")
    
    # Sort genres by frequency in descending order
    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("Genre Counts (Sorted by Frequency):")
    for genre, count in sorted_genres:
        print(f"{genre}: {count}")

if __name__ == "__main__":
    main()


# Genres that we will merge: 
# Electronic, Ambient Electronic, techno
# Rock, Indie-Rock, Psych-Rock
# Hip-hop
# Punk
# Folk
# Instrumental
# Pop, experimental pop
# Classical
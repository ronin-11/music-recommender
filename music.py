import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read the data
df = pd.read_csv("Spotify_final_dataset.csv", low_memory=False)[:1000]

# Preprocess the song names to lowercase
df['Song Name'] = df['Song Name'].str.lower()

# Remove duplicates
df = df.drop_duplicates(subset="Song Name")

# Drop null values
df = df.dropna(axis=0)

# Drop non-required columns
df = df.drop(df.columns[3:], axis=1)

# Removing space from "Artist Name" column
df["Artist Name"] = df["Artist Name"].str.replace(" ", "")

# Combine all columns and assign as new column
df["data"] = df.apply(lambda value: " ".join(value.astype("str")), axis=1)

# Models
vectorizer = CountVectorizer()
vectorized = vectorizer.fit_transform(df["data"])
similarities = cosine_similarity(vectorized)

# Assign the new dataframe with `similarities` values
df_tmp = pd.DataFrame(similarities, columns=df["Song Name"], index=df["Song Name"]).reset_index()

# Streamlit app
def main():
    st.title("Music Recommendation System")

    st.write("Enter the name of a song in lower_case to get recommendations:")

    input_song = st.text_input("Song Name", "")

    if input_song:
        input_song = input_song.strip().lower()  # Convert input to lowercase

        if input_song in df_tmp.columns:
            recommendation = df_tmp.nlargest(11, input_song)["Song Name"].values.tolist()
            recommendation.remove(input_song)  # Remove the input song from recommendations
            st.write("You should check out these songs:")
            for song in recommendation:
                st.write(song)
        else:
            st.write("Sorry, there is no song name in our database. Please try another one.")

if __name__ == "__main__":
    main()

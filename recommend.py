import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gdown
from io import BytesIO

class Recommender:
    def __init__(self):
        self.RATING_URL = "https://drive.google.com/uc?export=download&id=15Kz6hivg7pGtYvWoxb2Fw02N7Y0jbroo"
        self.USER_URL = "https://drive.google.com/uc?export=download&id=15rDWP29mhWwgAg_q74rLRfwpXSAiJAYu"
        self.BOOK_URL = "https://drive.google.com/uc?export=download&id=1avgdfmSp3-rf62UzrU8o5F4oeCAqp7ly"
        self.df_context, self.user_item_matrix, self.user_mapping, self.book_mapping, self.active_users = self.load_and_process_data()

    def download_and_read_zip(self, url):
        output = BytesIO()
        gdown.download(url, output, quiet=True)
        output.seek(0)
        return pd.read_csv(output, compression="zip", low_memory=False)

    def load_and_process_data(self):
        try:
            df_rating = self.download_and_read_zip(self.RATING_URL)
            df_user = self.download_and_read_zip(self.USER_URL)
            df_book = self.download_and_read_zip(self.BOOK_URL)

            active_users = df_rating["User-ID"].value_counts()
            active_users = active_users[active_users >= 20].index
            df_filtered = df_rating[df_rating["User-ID"].isin(active_users)]

            popular_books = df_filtered["ISBN"].value_counts()
            popular_books = popular_books[popular_books >= 20].index
            df_filtered = df_filtered[df_filtered["ISBN"].isin(popular_books)]

            df_filtered = df_filtered.merge(df_user[["User-ID", "Location"]], on="User-ID", how="left")
            df_filtered = df_filtered.merge(
                df_book[["ISBN", "Book-Title", "Image-URL-M"]], 
                on="ISBN", 
                how="left"
            )

            user_mapping = {id: idx for idx, id in enumerate(df_filtered["User-ID"].unique())}
            book_mapping = {isbn: idx for idx, isbn in enumerate(df_filtered["ISBN"].unique())}
            df_filtered["UserID"] = df_filtered["User-ID"].map(user_mapping)
            df_filtered["BookID"] = df_filtered["ISBN"].map(book_mapping)
            df_filtered["Rating"] = df_filtered["Book-Rating"]

            user_item_matrix = df_filtered.pivot(index="UserID", columns="BookID", values="Rating").fillna(0)

            return df_filtered, user_item_matrix, user_mapping, book_mapping, active_users
        except Exception as e:
            raise Exception(f"Error loading data: {e}")

    def recommend_books(self, user_id, num_books=10):
        try:
            if user_id not in self.user_mapping:
                valid_ids = list(self.active_users[:10])
                return None, f"User-ID {user_id} không hợp lệ. Vui lòng chọn User-ID trong khoảng: {valid_ids}..."

            user_index = self.user_mapping[user_id]

            m, n = self.user_item_matrix.shape
            k = int(np.sqrt(min(m, n)))
            svd = TruncatedSVD(n_components=k)
            U = svd.fit_transform(self.user_item_matrix)
            Sigma = svd.singular_values_
            VT = svd.components_

            r = self.user_item_matrix.iloc[user_index].values.reshape(1, -1)
            r_reduced = r[:, :VT.shape[1]]
            Unew = np.dot(r_reduced, VT.T) / Sigma
            similarities = cosine_similarity(Unew, U).flatten()
            top_10_users = [user for user in similarities.argsort()[::-1] if user != user_index][:10]

            test_user_rated_books = set(self.user_item_matrix.iloc[user_index].to_numpy().nonzero()[0])
            recommended_books = []
            seen_books = set()

            for user in top_10_users:
                user_top_books = self.df_context[
                    (self.df_context["UserID"] == user) & (self.df_context["Rating"] > 3)
                ].sort_values(by="Rating", ascending=False)

                for _, row in user_top_books.iterrows():
                    book_id = row["BookID"]
                    if book_id not in seen_books and book_id not in test_user_rated_books:
                        image_url = row["Image-URL-M"] if pd.notna(row["Image-URL-M"]) else "https://via.placeholder.com/100"
                        recommended_books.append({
                            "BookID": book_id,
                            "Book-Title": row["Book-Title"],
                            "Rating": row["Rating"],
                            "Image-URL": image_url
                        })
                        seen_books.add(book_id)

                    if len(recommended_books) >= num_books:
                        break
                if len(recommended_books) >= num_books:
                    break

            recommended_books.sort(key=lambda x: x["Rating"], reverse=True)
            return recommended_books, None
        except Exception as e:
            return None, f"Error in recommendation: {e}"
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from utils.text_processor import TextProcessor
#description


class Recomender:
    def __init__(self, df_train):
        self.df_train = df_train.copy()
        self.tfidf = TfidfVectorizer()
        self.text_processor = TextProcessor()
        self.cosine_sim = None

    def train_recomender(self):
        self.df_train['process_description'] = self.df_train['name'].apply(self.text_processor.text_process_recomender)#cambiar por descripcion qndo esten todas
        tfidf_matrix = self.tfidf.fit_transform(self.df_train['process_description'])
        if self.cosine_sim == None :
            self.cosine_sim = linear_kernel(tfidf_matrix,tfidf_matrix)


    def get_recommendations(self, names):
        recommendations = set()

        for name in names:
            try:
                idx = self.df_train[self.df_train['name'] == name].index[
                    0]
                sim_scores = list(enumerate(self.cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:3]
                item_index = [i[0] for i in sim_scores]
                recommendations.update(item_index)

            except IndexError:
                print(f"No se encontró la película: {name}")

        recommended_df = self.df_train.iloc[list(recommendations)].reset_index(drop=True)

        return recommended_df[['id', 'name', 'current_price','tag','shop_id']]




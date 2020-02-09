from sklearn.cluster import KMeans
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.neighbors import DistanceMetric
from ciu import determine_ciu

from music_data_generator import generate_data

dist = DistanceMetric.get_metric('euclidean')


class Predictor:
    def __init__(self, model, data, original_case):
        self.model = model
        self.data = data
        self.original_case = original_case

    def predict(self, cases):
        actual_prediction = self.model.predict(self.original_case)[0]
        distances = []
        for transformation in self.model.transform(cases):
            distances.append(-transformation[actual_prediction])
        print(distances)
        return distances


data = generate_data()
training_data = data['train'][1]
k_means = KMeans(n_clusters=5, random_state=0).fit(training_data)

test_data = data['test'][1]

predictor = Predictor(k_means, training_data, test_data[:1])
example_prediction = k_means.score(test_data[:1])

category_mapping = {
    'genre': ['genre_classic', 'genre_hiphop', 'genre_jazz', 'genre_pop',
              'genre_rock', 'genre_soul']
}

ciu = determine_ciu(
    test_data.iloc[0, :].to_dict(),
    predictor.predict,
    {
        'length': [90, 300, True],
        'volume': [0, 10, True],
        'danceability': [0, 10, True],
        'bpm': [60, 220, True],
        'year': [1950, 2020, True],
        'genre_classic': [0, 1, True],
        'genre_hiphop': [0, 1, True],
        'genre_jazz': [0, 1, True],
        'genre_pop': [0, 1, True],
        'genre_rock': [0, 1, True],
        'genre_soul': [0, 1, True]
    },
    100,
    None,
    category_mapping
)

print(data['test'][0].values[0])
print(test_data.values[0])
print(k_means.predict([test_data.values[0]]))

print(ciu.ci, ciu.cu)

ciu.plot_ci()
#ciu.plot_cu()

print(ciu.text_explain())

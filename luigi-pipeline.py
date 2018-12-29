# Create a pipeline using Luigi to predict sentiment of tweets with a logistic regression model.

import luigi
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# Tasks are run with the following:
# python luigi_pipeline.py CleanDataTask --local-scheduler


class CleanDataTask(luigi.Task):
    """ Cleans the input CSV file by removing any rows without valid geo-coordinates.

        Output file should contain just the rows that have geo-coordinates and
        non-(0.0, 0.0) files.
    """
    tweet_file = luigi.Parameter(default='airline_tweets.csv')
    output_file = luigi.Parameter(default='clean_data.csv')

    def requires(self):
        return []

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        tweets = pd.read_csv(self.tweet_file, encoding='iso-8859-1')  # shape: (14640, 20)

        empty_tweet_coord = tweets[tweets['tweet_coord'].isnull() == True].index  # tweets with empty coordinates
        zero_coord = tweets[tweets['tweet_coord'] == "[0.0, 0.0]"].index  # tweets with coordinates "[0.0, 0.0]"

        # Drop rows without valid geo-coordinates
        tweets = tweets.drop(index=empty_tweet_coord)
        tweets = tweets.drop(index=zero_coord)

        # Format coordinate columns
        tweets["tweet_coord"] = tweets["tweet_coord"].str.replace("[", "")
        tweets["tweet_coord"] = tweets["tweet_coord"].str.replace("]", "")

        # Extract latitude and longitude
        tweets["latitude"] = tweets["tweet_coord"].str.split(',').str[0].astype(float)
        tweets["longitude"] = tweets["tweet_coord"].str.split(',').str[1].astype(float)

        # Write clean tweets to output file
        tweets.to_csv(self.output_file, sep=',')


class TrainingDataTask(luigi.Task):

    """ Extracts features/outcome variable in preparation for training a model.

        Output file should have columns corresponding to the training data:
        - y = airline_sentiment (coded as 0=negative, 1=neutral, 2=positive)
        - X = a one-hot coded column for each city in "cities.csv"
    """
    tweet_file = luigi.Parameter(default='clean_data.csv')
    cities_file = luigi.Parameter(default='cities.csv')
    output_file = luigi.Parameter(default='features.csv')
    reference_cities = luigi.Parameter(default='closest_cities.pkl')

    def requires(self):
        return CleanDataTask(self.tweet_file)

    def output(self):
            return luigi.LocalTarget(self.output_file), luigi.LocalTarget(self.reference_cities)

    def run(self):
        tweets = pd.read_csv(self.tweet_file, encoding='iso-8859-1')
        cities = pd.read_csv(self.cities_file)

        target_column = "airline_sentiment"  # define the target
        sentiment_dict = {"negative": 0, "neutral": 1, "positive": 2}

        # Encode the target from categorical to numeric
        tweets[target_column] = tweets[target_column].map(sentiment_dict)

        # Drop extraneous columns in cities dataframe
        cities = cities.filter(['name', 'latitude', 'longitude', 'country code', 'timezone'])

        # Create coordinate pair columns
        tweets['coord'] = list(zip(tweets['latitude'], tweets['longitude']))
        city_coord = cities[['latitude', 'longitude']].values

        # List of closest city per tweet
        closest = []

        def nearest_city(tweet_coord, city_list):
            dist = np.sum((city_list - tweet_coord) ** 2, axis=1)
            return np.argmin(dist)

        # Find the closest city per tweet coordinates
        for i in range(tweets.shape[0]):
            closest_city_index = nearest_city(np.asarray(tweets.iloc[i]['coord']), city_coord)
            closest.append(cities['name'].loc[closest_city_index])

        tweets['closest_city'] = closest

        # Save the closest cities into a reference file
        pkl.dump(tweets['closest_city'], open('closest_cities.pkl', 'wb'))

        # One-hot encode city names and split into X, y datasets
        X = pd.get_dummies(tweets['closest_city'])
        y = tweets[target_column]

        # Prepare training dataset
        df = pd.concat([X, y], axis=1)

        # Write training dataset to output file
        df.to_csv(self.output_file, sep=',')


class TrainModelTask(luigi.Task):
    """ Trains a classifier to predict negative, neutral, positive
        based only on the input city.

        Output file should be the pickle'd model.
    """
    tweet_file = luigi.Parameter(default='features.csv')
    output_file = luigi.Parameter(default='model.pkl')

    def requires(self):
        return TrainingDataTask(self.tweet_file), TrainingDataTask(self.output_file)

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        df = pd.read_csv(self.tweet_file)

        # Set the target column
        y = df['airline_sentiment']

        # Set the feature matrix; one-hot encoded cities
        X = df.drop(columns='airline_sentiment')

        # Split training data into 70% training, 30% testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Fit a one-vs-all logistic regression classifier
        clf = linear_model.LogisticRegression()
        clf.fit(X_train, y_train)

        # Serialize trained model to output file
        pkl.dump(clf, open(self.output_file, 'wb'))


class ScoreTask(luigi.Task):
    """ Uses the scored model to compute the sentiment for each city.

        Output file should be a four column CSV with columns:
        - city name
        - negative probability
        - neutral probability
        - positive probability
    """
    tweet_file = luigi.Parameter(default='features.csv')
    output_file = luigi.Parameter(default='scores.csv')
    reference_cities = luigi.Parameter(default='closest_cities.pkl')

    def requires(self):
        return [TrainingDataTask(self.tweet_file), TrainModelTask(self.tweet_file), TrainingDataTask(self.output_file)]

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        df = pd.read_csv(self.tweet_file)

        # Get all feature rows to be scored, except for the target column
        X = df.drop(columns='airline_sentiment')

        # Instantiate the trained model
        with open('model.pkl', 'rb') as pickled_model:
            clf = pkl.load(pickled_model)

        # Instantiate the reference file for closest city per tweet
        with open('closest_cities.pkl', 'rb') as reference_cities:
            reference_cities = pd.DataFrame(pkl.load(reference_cities))

        # Use model to predict the probability of a negative, neutral, or positive sentiment class
        pred_prob = clf.predict_proba(X)

        # Join the closest city per tweet with the sentiment predictions from the model
        # Sort predictions dataframe by highest predicted positive sentiment
        sentiment_score = pd.DataFrame({'city': reference_cities['closest_city'],
                                        'negative_score': pred_prob[:, 0],
                                        'neutral_score': pred_prob[:, 1],
                                        'positive_score': pred_prob[:, 2]}) \
            .sort_values(by='positive_score', ascending=False)

        # Check for duplicates across rows, if any
        sentiment_score.drop_duplicates(inplace=True)

        # Save the sentiment-per-city dataframe to the output file
        sentiment_score.to_csv("scores.csv", sep=',')


if __name__ == "__main__":
    luigi.run()

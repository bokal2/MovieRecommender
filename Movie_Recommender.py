# Requires Numpy and scikit-surprise installed
from surprise import Reader, Dataset, SVD, evaluate


# Read data into an array of strings
with open('./ml-100k/u.data') as f:
    all_lines = f.readlines()

# Let's prepare data to be used in Surprise
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file('./ml-100k/u.data', reader=reader)

# We Split the dataset into 5 folds and choose the algorithm
data.split(n_folds=5)
algo = SVD() # our chosen algorithm

# We now train and test reporting the RMSE and MAE scores
evaluate(algo, data, measures=['RMSE', 'MAE'])

# Retrieving the trainset.
trainset = data.build_full_trainset()
algo.train(trainset)

# Predict a sample item
userid = str(196)
itemid = str(302)
actual_rating = 4

#Printing out our predictions
print(algo.predict(userid, itemid, actual_rating))

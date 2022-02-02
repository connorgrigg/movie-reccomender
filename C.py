import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')

data = ratings.pivot(index='movieId', columns='userId', values='rating')
data.fillna(0, inplace=True)
csr = csr_matrix(data.values)
data.reset_index(inplace=True)

knn = NearestNeighbors(metric='cosine', algorithm='auto')
knn.fit(csr)

user_count = ratings.groupby('movieId')['rating'].agg('count')
movie_count = ratings.groupby('userId')['rating'].agg('count')
user_average_vote = ratings.groupby('userId')['rating'].agg('mean')

first = plt.figure(1)
first.canvas.manager.set_window_title('Movie Vote Counts')
plt.title('How many votes each Movie ID received')
plt.scatter(user_count.index, user_count, color='blue')
plt.xlabel('Movie ID')
plt.ylabel('Vote count')

second = plt.figure(2)
second.canvas.manager.set_window_title('User Vote Counts')
plt.title('Votes per User ID')
plt.scatter(movie_count.index, movie_count, color='blue')
plt.xlabel('User ID')
plt.ylabel('How many movies each User ID voted on')

third = plt.figure(3)
third.canvas.manager.set_window_title('Average User Vote')
plt.title('Average user movie rating')
plt.scatter(user_average_vote.index, user_average_vote, color='blue')
plt.xlabel('User ID')
plt.ylabel('Average Vote')

plt.show()


def get_movies(inp):
    ret = []
    ret_count = 5
    ml = movies[movies['title'].str.contains(inp)]
    if len(ml):
        movie_index = ml.iloc[0]['movieId']
        movie_index = data[data['movieId'] == movie_index].index[0]
        distances, indices = knn.kneighbors(csr[movie_index], n_neighbors=ret_count + 1)
        rec = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
        for val in rec:
            movie_index = data.iloc[val[0]]['movieId']
            index = movies[movies['movieId'] == movie_index].index
            ret.append({'Title': movies.iloc[index]['title'].values[0]})
        ret = pd.DataFrame(ret, index=range(1, ret_count + 1))
        return ret
    else:
        return "Invalid input"


while True:
    try:
        answer = int(input("1. Search a movie\n2. Quit\n"))
    except:
        print("Invalid input, please pick 1 or 2")
        continue
    if answer == 2:
        break
    elif answer == 1:
        inp = input("Open file /data/movies.csv, pick a movie to search, and enter the case sensitive title without "
                    "the year\nExample input: Fight Club\n")
        if inp == "":
            inp = input("Please provide a query from the previous instructions\n")
        if inp == "":
            inp = input("Please provide a query from the initial instructions\n")
        if inp == "":
            inp = input("Please provide a query from the initial instructions\n")
        if inp == "":
            inp = input("Please provide a query from the initial instructions\n")
        if inp == "":
            inp = input("Please provide a query from the initial instructions, or general results will be returned\n")
        print(get_movies(inp), "\n")

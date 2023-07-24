from pyspark.context import SparkContext
import os
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from flask import Flask, render_template, request

app = Flask(__name__)

# Set the environment variables with double quotes
os.environ['PYSPARK_DRIVER_PYTHON'] = '"C:\\Users\\EBEN EZER NAPITU\\AppData\\Local\\Programs\\Python\\Python310\\python.exe"'
os.environ['PYSPARK_PYTHON'] = '"C:\\Users\\EBEN EZER NAPITU\\AppData\\Local\\Programs\\Python\\Python310\\python.exe"'

# Set the SPARK_HOME environment variable with the path to the Spark home directory
os.environ['SPARK_HOME'] = 'C:\\Users\\EBEN EZER NAPITU\\Downloads\\spark-3.1.2-bin-hadoop3.2\\spark-3.1.2-bin-hadoop3.2'

# Initialize the SparkSession
spark = SparkSession.builder \
    .appName("MyApp") \
    .master("local[1]") \
    .getOrCreate()

# Load the trained model
model_path = 'my_model'  # Replace with the actual path where your model is saved
Best_model = ALSModel.load(model_path)

# Endpoint for the home page
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Endpoint for movie recommendations
@app.route("/recommend", methods=["GET"])
def recommend_movies():
    user_id = int(request.args.get("user_id"))

    # Create a DataFrame with the user_id to get recommendations
    user_df = spark.createDataFrame([(user_id,)], ["userId"])
    recommendations = Best_model.recommendForUserSubset(user_df, 10)

    # Extract the movieId and rating from the recommendations
    movie_ids = [row.movieId for row in recommendations.collect()[0]["recommendations"]]
    ratings = [row.rating for row in recommendations.collect()[0]["recommendations"]]

    # Convert to a list of dictionaries
    recommended_movies = [{"movieId": movie_id, "rating": rating} for movie_id, rating in zip(movie_ids, ratings)]

    return render_template("recommend.html", recommendations=recommended_movies)

if __name__ == "__main__":
    app.run(debug=True)

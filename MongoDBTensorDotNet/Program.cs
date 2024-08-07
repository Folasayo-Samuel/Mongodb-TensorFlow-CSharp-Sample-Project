using MongoDB.Bson;
using MongoDB.Driver;
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using static System.Console;

class Program
{
    static void Main(string[] args)
    {
        // MongoDB connection string
        var client = new MongoClient("mongodb://localhost:27017");
        var database = client.GetDatabase("linear-data");
        var collection = database.GetCollection<BsonDocument>("sampleData");

        // Define the data to insert
        var sampleData = new SampleData
        {
            X = new List<double> { 3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1 },
            Y = new List<double> { 1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3 }
        };

        // Convert SampleData to BsonDocument
        var document = new BsonDocument
        {
            { "X", new BsonArray(sampleData.X) },
            { "Y", new BsonArray(sampleData.Y) }
        };

        // Insert the data into MongoDB
        collection.InsertOne(document);

        WriteLine("Data seeded successfully!");

        // Fetch the data from MongoDB
        var fetchedDocument = collection.Find(new BsonDocument()).FirstOrDefault();

        var X_array = fetchedDocument["X"].AsBsonArray.Select(value => (float)value.AsDouble).ToArray();
        var Y_array = fetchedDocument["Y"].AsBsonArray.Select(value => (float)value.AsDouble).ToArray();

        // Create a new ML context
        var mlContext = new MLContext();

        // Create the ML.NET data structures
        var data = X_array.Zip(Y_array, (x, y) => new DataPoint { X = x, Y = y }).ToList();
        var dataView = mlContext.Data.LoadFromEnumerable(data);

        // Define the trainer
        var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "X" })
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Y", featureColumnName: "Features"));

        // Train the model
        var model = pipeline.Fit(dataView);

        // Use the model to make predictions
        var predictions = model.Transform(dataView);
        var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Y");

        WriteLine($"R^2: {metrics.RSquared}");
        WriteLine($"RMSE: {metrics.RootMeanSquaredError}");

        // Display the predictions
        var predictionFunction = mlContext.Model.CreatePredictionEngine<DataPoint, Prediction>(model);
        foreach (var point in data)
        {
            var prediction = predictionFunction.Predict(point);
            WriteLine($"X: {point.X}, Y: {point.Y}, Predicted: {prediction.PredictedY}");
        }
    }

     public class SampleData
    {
        public List<double> X { get; set; } = new List<double>();
        public List<double> Y { get; set; } = new List<double>();
    }

    public class DataPoint
    {
        public float X { get; set; }
        public float Y { get; set; }
    }

    public class Prediction
    {
        [ColumnName("Score")]
        public float PredictedY { get; set; }
    }
}

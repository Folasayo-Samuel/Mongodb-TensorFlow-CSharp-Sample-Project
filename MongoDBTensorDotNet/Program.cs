using MongoDB.Bson;
using MongoDB.Driver;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow;
using Tensorflow.NumPy;
using System;
using System.Linq;

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

        Console.WriteLine("Data seeded successfully!");

        // Fetch the data from MongoDB
        var fetchedDocument = collection.Find(new BsonDocument()).FirstOrDefault();

        var X_array = fetchedDocument["X"].AsBsonArray.Select(value => (float)value.AsDouble).ToArray();
        var Y_array = fetchedDocument["Y"].AsBsonArray.Select(value => (float)value.AsDouble).ToArray();

        // Convert arrays to NumPy arrays
        var X = np.array(X_array);
        var Y = np.array(Y_array);
        var n_samples = X.shape[0];

        // Parameters        
        var training_steps = 1000;
        var learning_rate = 0.01f;
        var display_step = 100;

        // We can set a fixed init value in order to demo
        var W = tf.Variable(-0.06f, name: "weight");
        var b = tf.Variable(-0.73f, name: "bias");
        var optimizer = keras.optimizers.SGD(learning_rate);

        // Run training for the given number of steps.
        foreach (var step in range(1, training_steps + 1))
        {
            // Run the optimization to update W and b values.
            // Wrap computation inside a GradientTape for automatic differentiation.
            using var g = tf.GradientTape();
            // Linear regression (Wx + b).
            var pred = W * X + b;
            // Mean square error.
            var loss = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples);
            // should stop recording
            // Compute gradients.
            var gradients = g.gradient(loss, (W, b));

            // Update W and b following gradients.
            optimizer.apply_gradients(zip(gradients, (W, b)));

            if (step % display_step == 0)
            {
                pred = W * X + b;
                loss = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples);
                print($"step: {step}, loss: {loss.numpy()}, W: {W.numpy()}, b: {b.numpy()}");
            }
        }
    }
}

public class SampleData
{
    public List<double> X { get; set; }
    public List<double> Y { get; set; }
}

// class Program
// {
// 	static void Main(string[] args)
// 	{
// 		// MongoDB connection string
// 		var client = new MongoClient("mongodb://localhost:27017");
// 		var database = client.GetDatabase("yourDatabase");
// 		var collection = database.GetCollection<BsonDocument>("sampleData");
		
// 		// Define the data to insert
// var sampleData = new SampleData
// {
// 	X = new List<double> { 3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1 },
// 	Y = new List<double> { 1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3 }
// };


// // Define the data to insert
// 		// var sampleData = new BsonDocument
// 		// {
// 		//     { "X", new BsonArray { 3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1 } },
// 		//     { "Y", new BsonArray { 1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3 } }
// 		// };

// 		// Insert the data into MongoDB
// 		collection.InsertOne(sampleData);

// 		Console.WriteLine("Data seeded successfully!");	

// 		// Fetch the data from MongoDB
// var document = collection.Find(new BsonDocument()).FirstOrDefault();

// var X_array = document["X"].AsBsonArray.Select(value => (float)value.AsDouble).ToArray();
// var Y_array = document["Y"].AsBsonArray.Select(value => (float)value.AsDouble).ToArray();
	
		
// 		// Fetch the data from MongoDB
// 		// var document = collection.Find(new BsonDocument()).FirstOrDefault();

// 		// var X_array = document["X"].AsBsonArray.Select(value => (float)value.AsDouble).ToArray();
// 		// var Y_array = document["Y"].AsBsonArray.Select(value => (float)value.AsDouble).ToArray();

// 		// Convert arrays to NumPy arrays
// 		var X = np.array(X_array);
// 		var Y = np.array(Y_array);
// 		var n_samples = X.shape[0];

// 		// Parameters        
// 		var training_steps = 1000;
// 		var learning_rate = 0.01f;
// 		var display_step = 100;

// 		// We can set a fixed init value in order to demo
// 		var W = tf.Variable(-0.06f, name: "weight");
// 		var b = tf.Variable(-0.73f, name: "bias");
// 		var optimizer = keras.optimizers.SGD(learning_rate);

// 		// Run training for the given number of steps.
// 		foreach (var step in range(1, training_steps + 1))
// 		{
// 			// Run the optimization to update W and b values.
// 			// Wrap computation inside a GradientTape for automatic differentiation.
// 			using var g = tf.GradientTape();
// 			// Linear regression (Wx + b).
// 			var pred = W * X + b;
// 			// Mean square error.
// 			var loss = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples);
// 			// should stop recording
// 			// Compute gradients.
// 			var gradients = g.gradient(loss, (W, b));

// 			// Update W and b following gradients.
// 			optimizer.apply_gradients(zip(gradients, (W, b)));

// 			if (step % display_step == 0)
// 			{
// 				pred = W * X + b;
// 				loss = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples);
// 				print($"step: {step}, loss: {loss.numpy()}, W: {W.numpy()}, b: {b.numpy()}");
// 			}
// 		}
// 	}
// }

// public class SampleData
// {
// 	public List<double> X { get; set; }
// 	public List<double> Y { get; set; }
// }

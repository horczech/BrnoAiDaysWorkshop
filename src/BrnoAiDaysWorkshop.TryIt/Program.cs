using BrnoAiDaysWorkshop.Training;
using Microsoft.ML;
using OpenCvSharp;

var mlContext = new MLContext();
var modelPath = "..\\..\\..\\..\\BrnoAiDaysWorkshop\\bin\\Debug\\net6.0\\model\\trained_model";
if (!File.Exists(modelPath)) {
    Console.WriteLine("Model is not trained yet! Make sure to run training first.");
    return;
}
var model = mlContext.Model.Load(modelPath, out _);
var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageData, PredictionModel>(model);

AskForInput();
string? filePath;
while ((filePath = Console.ReadLine()) != "exit") {
    try {
        if (filePath is null) {
            Console.WriteLine("Path cannot be empty");
            continue;
        }

        FileInfo fileInfo;
        try {
            fileInfo = new FileInfo(filePath.Trim('"'));
            if (!fileInfo.Exists) {
                Console.WriteLine("This file does not exist, entry full path of the image file including extension.");
                continue;
            }
        }
        catch (Exception) {
            Console.WriteLine("Invalid file path.");
            continue;
        }

        var imageData = new ImageData { ImagePath = fileInfo.FullName, Image = Cv2.ImRead(fileInfo.FullName).ToBytes()};
        var prediction = predictionEngine.Predict(imageData);
        Console.WriteLine($"Image: {imageData.GetFileName()} | Predicted Value: {prediction.PredictedLabel}");
    }
    finally {
        AskForInput();
    }
}

void AskForInput() {
    Console.Write("\nInput full path of the image OR type 'exit' to end the application:\nFile path:");
}

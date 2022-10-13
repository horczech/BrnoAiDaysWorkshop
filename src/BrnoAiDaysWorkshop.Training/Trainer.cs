using System.Drawing;
using Microsoft.ML;
using Microsoft.ML.Transforms;
using Microsoft.ML.Vision;
using OpenCvSharp;

namespace BrnoAiDaysWorkshop.Training;

public static class Trainer
{
    private const string ModelFolder = "model";
    private const string TrainedModelPath = $"{ModelFolder}/trained_model";

    // for reference, go to https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/image-classification-api-transfer-learning
    public static (FileInfo trainedModelPath, IDataView testSet) Train(string imagesFolder) {
        Directory.CreateDirectory(ModelFolder);
        var mlContext = new MLContext();
        var aaa = new FileInfo(TrainedModelPath);
        var files = Directory
            .EnumerateFiles(imagesFolder, "*", SearchOption.AllDirectories).Select(x => new FileInfo(x))
            .Select(x => new ImageData { ImagePath = x.FullName, Label = x.Directory!.Name, Image = Cv2.ImRead(x.FullName).ToBytes() }).ToList();
        var data = mlContext.Data.LoadFromEnumerable(files);
        var shuffledData = mlContext.Data.ShuffleRows(data);
        var preprocessingPipeline = mlContext.Transforms.Conversion
            .MapValueToKey(inputColumnName: nameof(ImageData.Label), outputColumnName: nameof(ImageData.LabelAsKey), keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue);
        var preprocessedData = preprocessingPipeline.Fit(shuffledData).Transform(shuffledData);

        var trainSplit = mlContext.Data.TrainTestSplit(data: preprocessedData, testFraction: 0.2);
        var validationTestSplit = mlContext.Data.TrainTestSplit(trainSplit.TestSet, testFraction: 0.5); //ToDo M: change to 0.2 when image image augumentation (evaluation is not getting any images when fraction is 0.1)

        //ToDo M: extraxt train/test splitting out? its super weird that Train method returns test set
        var trainSet = trainSplit.TrainSet;
        var validationSet = validationTestSplit.TrainSet;
        var testSet = validationTestSplit.TestSet;

        var trainingPipeline = mlContext.MulticlassClassification.Trainers
            .ImageClassification(CreateOptions(validationSet))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue(nameof(PredictionModel.PredictedLabel)));

        var trainedModel = trainingPipeline.Fit(trainSet);

        var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageData, PredictionModel>(trainedModel);
        foreach (var imageData in files) {
            var predictionModel = predictionEngine.Predict(imageData);
            Console.WriteLine($"File: {imageData.ImagePath} -> {(predictionModel.PredictedLabel == imageData.Label ? "Correct" : "Wrong!!!")}");
        }

        mlContext.Model.Save(trainedModel, preprocessedData.Schema, TrainedModelPath);
        return (new FileInfo(TrainedModelPath), testSet);
    }

    private static ImageClassificationTrainer.Options CreateOptions(IDataView validationSet) =>
        new ImageClassificationTrainer.Options()
        {
            FeatureColumnName = nameof(ImageData.Image),
            LabelColumnName = "LabelAsKey",
            Arch = ImageClassificationTrainer.Architecture.MobilenetV2,
            BatchSize = 8,
            LearningRate = 0.01f,
            MetricsCallback = Console.WriteLine,
            ValidationSet = validationSet,
            EarlyStoppingCriteria = new ImageClassificationTrainer.EarlyStopping(0.001f, 3)
        };
}
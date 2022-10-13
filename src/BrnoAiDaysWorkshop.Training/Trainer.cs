using Microsoft.ML;
using Microsoft.ML.Vision;

namespace BrnoAiDaysWorkshop.Training;

public static class Trainer
{
    private const string ModelFolder = "model";
    private const string TrainedModelPath = $"{ModelFolder}/trained_model";

    // for reference, go to https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/image-classification-api-transfer-learning
    public static (FileInfo trainedModelPath, IDataView testSet) Train(string imagesFolder) {
        Directory.CreateDirectory(ModelFolder);
        var mlContext = new MLContext();

        var files = Directory
            .EnumerateFiles(imagesFolder, "*", SearchOption.AllDirectories).Select(x => new FileInfo(x))
            .Select(x => new ImageData { ImagePath = x.FullName, Label = x.Directory!.Name });
        var data = mlContext.Data.LoadFromEnumerable(files);
        var shuffledData = mlContext.Data.ShuffleRows(data);
        var preprocessingPipeline = mlContext.Transforms.Conversion
            .MapValueToKey(inputColumnName: nameof(ImageData.Label), outputColumnName: nameof(ImageData.LabelAsKey))
            .Append(mlContext.Transforms.LoadRawImageBytes(outputColumnName: nameof(ImageData.Image), imageFolder: imagesFolder, inputColumnName: nameof(ImageData.ImagePath)));
        var preprocessedData = preprocessingPipeline.Fit(shuffledData).Transform(shuffledData);

        var trainSplit = mlContext.Data.TrainTestSplit(data: preprocessedData, testFraction: 0.2);
        var validationTestSplit = mlContext.Data.TrainTestSplit(trainSplit.TestSet);

        var trainSet = trainSplit.TrainSet;
        var validationSet = validationTestSplit.TrainSet;
        var testSet = validationTestSplit.TestSet;

        var trainingPipeline = mlContext.MulticlassClassification.Trainers
            .ImageClassification(CreateOptions(validationSet))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
        var trainedModel = trainingPipeline.Fit(trainSet);

        mlContext.Model.Save(trainedModel, preprocessedData.Schema, TrainedModelPath);
        return (new FileInfo(TrainedModelPath), testSet);
    }

    private static ImageClassificationTrainer.Options CreateOptions(IDataView validationSet) {
        return new ImageClassificationTrainer.Options
        {
            FeatureColumnName = nameof(ImageData.Image),
            LabelColumnName = nameof(ImageData.LabelAsKey),
            ValidationSet = validationSet,
            Arch = ImageClassificationTrainer.Architecture.MobilenetV2,
            MetricsCallback = Console.WriteLine,
            TestOnTrainSet = false,
            ReuseTrainSetBottleneckCachedValues = true,
            ReuseValidationSetBottleneckCachedValues = true
        };
    }
}
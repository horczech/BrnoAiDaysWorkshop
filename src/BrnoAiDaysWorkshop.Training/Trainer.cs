using Microsoft.ML;
using Microsoft.ML.Transforms;
using Microsoft.ML.Vision;

namespace BrnoAiDaysWorkshop.Training;

public static class Trainer
{
    private const string ModelFolder = "model";
    private const string TrainedModelPath = $"{ModelFolder}/trained_model";

    // for reference, go to https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/image-classification-api-transfer-learning
    public static FileInfo Train(IDataView trainData, string imageFolder) {
        Directory.CreateDirectory(ModelFolder);
        var mlContext = new MLContext();

        var preprocessingPipeline = mlContext.Transforms.Conversion
            .MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelAsKey", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
            .Append( mlContext.Transforms.LoadRawImageBytes(imageFolder: imageFolder, inputColumnName: "ImagePath", outputColumnName: "Feature"));

        var preprocessedData = preprocessingPipeline.Fit(trainData).Transform(trainData);
        var validationTestSplit = mlContext.Data.TrainTestSplit(preprocessedData, testFraction: 0.5);
        var trainSet = validationTestSplit.TrainSet;
        var validationSet = validationTestSplit.TestSet;

        var trainingPipeline = mlContext.MulticlassClassification.Trainers
            .ImageClassification(CreateOptions(validationSet))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"))
            .AppendCacheCheckpoint(mlContext);

        var trainedModel = preprocessingPipeline.Append(trainingPipeline).Fit(trainSet);

        mlContext.Model.Save(trainedModel, preprocessedData.Schema, TrainedModelPath);
        return new FileInfo(TrainedModelPath);

    }

    private static ImageClassificationTrainer.Options CreateOptions(IDataView validationSet) =>
        new ImageClassificationTrainer.Options()
        {
            FeatureColumnName = "Feature",
            LabelColumnName = "LabelAsKey",
            Arch = ImageClassificationTrainer.Architecture.MobilenetV2,
            BatchSize = 8,
            LearningRate = 0.01f,
            MetricsCallback = Console.WriteLine,
            ValidationSet = validationSet,
            EarlyStoppingCriteria = new ImageClassificationTrainer.EarlyStopping(0.001f, 3)
        };
}
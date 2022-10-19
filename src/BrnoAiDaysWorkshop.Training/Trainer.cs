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

        // todo: TASK 2a Fill Input and Output column in value-key mapping
        var preprocessingPipeline = mlContext.Transforms.Conversion
            .MapValueToKey("", "")
            .Append( mlContext.Transforms.LoadRawImageBytes(imageFolder: imageFolder, inputColumnName: "ImagePath", outputColumnName: "Feature"));

        // todo: TASK 2b apply preprocessing pipeline to the train data
        IDataView preprocessedData = null;
        var validationTestSplit = mlContext.Data.TrainTestSplit(preprocessedData, testFraction: 0.5);
        var trainSet = validationTestSplit.TrainSet;
        var validationSet = validationTestSplit.TestSet;

        var trainingPipeline = mlContext.MulticlassClassification.Trainers
            .ImageClassification(CreateOptions(validationSet))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"))
            .AppendCacheCheckpoint(mlContext);

        var trainedModel = preprocessingPipeline.Append(trainingPipeline).Fit(trainSet);

        // todo: TASK 2d save model to the TrainedModelPath path

        return new FileInfo(TrainedModelPath);

    }

    private static ImageClassificationTrainer.Options CreateOptions(IDataView validationSet) =>
        new ImageClassificationTrainer.Options
        {
            // todo: TASK 2c create training options
        };
}
using BrnoAiDaysWorkshop.Training;
using ImageClassification.DataModels;
using Microsoft.ML;
using Microsoft.ML.Transforms;
using MnistClassification;

#region Paths
var datasetFolder = PathHelper.GetAbsolutePath(@"../../../Dataset");
var pretrainedModelPath = "../../../PretrainedModel/inception_v3_2016_08_28_frozen.pb";
var trainedModelPath = "../../../TrainedModel/Model.zip";
#endregion

var mlContext = new MLContext();

#region STEP 1: Load Data

Console.WriteLine("Loading dataset...");
var dataset = DatasetLoader.LoadImages(datasetFolder);
var shuffledDataset = mlContext.Data.ShuffleRows(dataset);
var datasetSplit = mlContext.Data.TrainTestSplit(data: shuffledDataset, testFraction: 0.2);

#endregion

#region STEP 2: Preprocess Data

var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelAsKey", inputColumnName: "Label")
    .Append(mlContext.Transforms.LoadImages(inputColumnName: "ImagePath", outputColumnName: "LoadedImage", imageFolder: datasetFolder))
    .Append(mlContext.Transforms.ResizeImages(inputColumnName: "LoadedImage", outputColumnName: "ResizedImage", imageWidth: ImageSettingsForTFModel.imageWidth, imageHeight: ImageSettingsForTFModel.imageHeight))
    .Append(mlContext.Transforms.ExtractPixels(inputColumnName: "ResizedImage", outputColumnName: "ExtractedPixels", interleavePixelColors: ImageSettingsForTFModel.channelsLast, offsetImage: ImageSettingsForTFModel.mean, scaleImage: ImageSettingsForTFModel.scale)) 
    .Append(mlContext.Model.LoadTensorFlowModel(pretrainedModelPath).ScoreTensorFlowModel(inputColumnNames: new[] { "ExtractedPixels" }, outputColumnNames: new[] { "InceptionV3/Predictions/Reshape" }, addBatchDimensionInput: false));

#endregion

#region STEP 3: Create training pipeline

var trainer = mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelAsKey", featureColumnName: "InceptionV3/Predictions/Reshape");
var trainingPipeline = dataProcessPipeline.Append(trainer)
    .Append(mlContext.Transforms.Conversion.MapKeyToValue(inputColumnName: "PredictedLabel", outputColumnName:"PredictedLabelValue"));

#endregion

#region STEP 4: Train & Save 
Console.WriteLine("Training the model...");
var trainedModel = trainingPipeline.Fit(datasetSplit.TrainSet);
mlContext.Model.Save(trainedModel, datasetSplit.TrainSet.Schema, trainedModelPath);

#endregion

#region STEP 5: Evaluate
Console.WriteLine("Evaluating the model...");
var predictions = trainedModel.Transform(datasetSplit.TestSet);
var metrics = mlContext.MulticlassClassification.Evaluate(data: predictions, labelColumnName: "LabelAsKey", scoreColumnName: "PredictedLabel");

Console.WriteLine($"Evaluation metrics for trained model");
Console.WriteLine($"AccuracyMacro = {metrics.MacroAccuracy:F4}");
Console.WriteLine($"AccuracyMicro = {metrics.MicroAccuracy:F4}");
Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

#endregion

Console.WriteLine("Aaaand we are done!!!!");

#region Helper Objects

public struct ImageSettingsForTFModel {
    public const int imageHeight = 299;
    public const int imageWidth = 299;
    public const float mean = 117;
    public const float scale = 1 / 255f;
    public const bool channelsLast = true;
};

#endregion



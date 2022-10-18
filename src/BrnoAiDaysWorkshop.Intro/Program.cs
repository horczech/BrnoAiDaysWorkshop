using BrnoAiDaysWorkshop.Intro.DataStructures;
using Microsoft.ML;
using Microsoft.ML.Transforms;
using MnistClassification;

#region Paths
var datasetPath = PathHelper.GetAbsolutePath(@"../../../Data/mnist-dataset.csv");
var modelRelativePath = "../../../MLModels/Model.zip";
var modelPath = PathHelper.GetAbsolutePath(modelRelativePath);
#endregion

var mlContext = new MLContext();

#region STEP 1: Load Data

var dataset = mlContext.Data.LoadFromTextFile<InputData>(datasetPath, separatorChar: ',', hasHeader: false);
var datasetSplit = mlContext.Data.TrainTestSplit(data: dataset, testFraction: 0.2);

#endregion

#region STEP 2: Preprocess Data

var dataProcessPipeline = mlContext.Transforms.Conversion
    .MapValueToKey(inputColumnName: "Number", outputColumnName: "Label", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
    .Append(mlContext.Transforms.Concatenate(inputColumnNames: "PixelValues", outputColumnName: "Features"))
    .AppendCacheCheckpoint(mlContext);

#endregion

#region STEP 3: Create training pipeline

var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");
var trainingPipeline = dataProcessPipeline.Append(trainer)
    .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "Number", inputColumnName: "Label"));

#endregion

#region STEP 4: Train & Save 

var trainedModel = trainingPipeline.Fit(datasetSplit.TrainSet);
mlContext.Model.Save(trainedModel, datasetSplit.TrainSet.Schema, modelPath);

#endregion

#region STEP 5: Evaluate

var predictions = trainedModel.Transform(datasetSplit.TestSet);
var metrics = mlContext.MulticlassClassification.Evaluate(data: predictions, labelColumnName: "Number", scoreColumnName: "Score");

Console.WriteLine($"Evaluation metrics for trained model");
Console.WriteLine($"AccuracyMacro = {metrics.MacroAccuracy:F4}");
Console.WriteLine($"AccuracyMicro = {metrics.MicroAccuracy:F4}");
Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

#endregion

#region STEP 6: Use trained model

var model = mlContext.Model.Load(modelPath, out _);
var predictionEngine = mlContext.Model.CreatePredictionEngine<InputData, OutputData>(model);
var prediction = predictionEngine.Predict(SampleData.Num9);

Console.WriteLine("\n\nPrediction scores:");
Console.WriteLine($"0: {prediction.Score[0]:F4}");
Console.WriteLine($"1: {prediction.Score[1]:F4}");
Console.WriteLine($"2: {prediction.Score[2]:F4}");
Console.WriteLine($"3: {prediction.Score[3]:F4}");
Console.WriteLine($"4: {prediction.Score[4]:F4}");
Console.WriteLine($"5: {prediction.Score[5]:F4}");
Console.WriteLine($"6: {prediction.Score[6]:F4}");
Console.WriteLine($"7: {prediction.Score[7]:F4}");
Console.WriteLine($"8: {prediction.Score[8]:F4}");
Console.WriteLine($"9: {prediction.Score[9]:F4}");

#endregion
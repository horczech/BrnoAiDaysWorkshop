// See https://aka.ms/new-console-template for more information
using BrnoAiDaysWorkshop.Evaluation;
using BrnoAiDaysWorkshop.Training;
using BrnoAiDaysWorkshop.Augmentation;
using OpenCvSharp;
using Microsoft.ML;

const string augmentedDataFolder = "augumented_data";
const string originalDataFolder = "data";
const bool generateDataset = true;

Console.WriteLine("Hello, World!");
var mlContext = new MLContext();

//STEP 1: Expand dataset
if (generateDataset) {
    Console.WriteLine("Clearing augmented data folder!");
    if (Directory.Exists(augmentedDataFolder))
        Directory.Delete(augmentedDataFolder, true);

    Directory.CreateDirectory(augmentedDataFolder + "/on");
    Directory.CreateDirectory(augmentedDataFolder + "/off");

    Console.WriteLine("Generating augmented data!");
    var originalImages = Directory
        .EnumerateFiles(originalDataFolder, "*", SearchOption.AllDirectories).Select(x => new FileInfo(x))
        .Select(x => new OriginalImageData { FileName = x.Name, FilePath = x.FullName, FileExtension = x.Extension, Label = x.Directory!.Name, Folder = augmentedDataFolder }).ToList();

    Augmentator.GenerateAugmentedDataSet(originalImages);
    Console.WriteLine("Augmentation data generated!");
}

//STEP 2: Load dataset
var dataset = DatasetLoader.LoadImages(augmentedDataFolder);
var splitDataset = mlContext.Data.TrainTestSplit(dataset, 0.2);

//STEP 3: Train model
Console.WriteLine("Model Training!");
var trainedModelPath = Trainer.Train(splitDataset.TrainSet, augmentedDataFolder);
Console.WriteLine("Model Trained!");

//STEP 4: Evaluate model
Console.WriteLine("Model Evaluation!");
Evaluator.Evaluate(trainedModelPath, splitDataset.TestSet);
Console.WriteLine("Model Evaluated!");

//STEP 5: Enjoy your model
Console.WriteLine("Bye, World!");




// See https://aka.ms/new-console-template for more information
using BrnoAiDaysWorkshop.Evaluation;
using BrnoAiDaysWorkshop.Training;
using BrnoAiDaysWorkshop.Augmentation;
using Microsoft.ML;

const string augmentedDataFolder = "augumented_data";
const string originalDataFolder = "data";
const bool generateDataset = false;

Console.WriteLine("Hello, World!");
var mlContext = new MLContext();

//todo: TASK 4: Expand dataset
if (generateDataset) {
    Console.WriteLine("Clearing augmented data folder!");
    if (Directory.Exists(augmentedDataFolder))
        Directory.Delete(augmentedDataFolder, true);

    Directory.CreateDirectory(augmentedDataFolder + "/on");
    Directory.CreateDirectory(augmentedDataFolder + "/off");

    Console.WriteLine("Generating augmented data!");
    var originalImages = Directory
        .EnumerateFiles(originalDataFolder, "*", SearchOption.AllDirectories).Select(x => new FileInfo(x))
        .Select(x => new OriginalImageData { FileName = x.Name.Replace(x.Extension, string.Empty), FilePath = x.FullName, FileExtension = x.Extension, Label = x.Directory!.Name, Folder = augmentedDataFolder }).ToList();

    Augmentator.GenerateAugmentedDataSet(originalImages);
    Console.WriteLine("Augmentation data generated!");
}

//Load dataset
var dataset = DatasetLoader.LoadImages(augmentedDataFolder);
// todo: TASK 1 split dataset into Train (80%) and Test(20%), save it into splitDataset local variable
dynamic? splitDataset = null;

//Train model
Console.WriteLine("Model Training!");
var trainedModelPath = Trainer.Train(splitDataset.TrainSet, augmentedDataFolder);
Console.WriteLine("Model Trained!");

//Evaluate model
Console.WriteLine("Model Evaluation!");
Evaluator.Evaluate(trainedModelPath, splitDataset.TestSet);
Console.WriteLine("Model Evaluated!");

//Enjoy your model
Console.WriteLine("Bye, World!");




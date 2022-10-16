// See https://aka.ms/new-console-template for more information
using BrnoAiDaysWorkshop.Evaluation;
using BrnoAiDaysWorkshop.Training;
using BrnoAiDaysWorkshop.Augmentation;
using OpenCvSharp;

const string augmentedDataFolder = "augmented_data";
const string originalDataFolder = "data";
const bool generateDataset = true;


Console.WriteLine("Hello, World!");

if (generateDataset) {
    Console.WriteLine("Clearing augmented data folder!");
    if (Directory.Exists(augmentedDataFolder))
        Directory.Delete(augmentedDataFolder, true);

    Directory.CreateDirectory(augmentedDataFolder + "/on");
    Directory.CreateDirectory(augmentedDataFolder + "/off");

    Console.WriteLine("Generating augmented data!");
    var originalImages = Directory
        .EnumerateFiles(originalDataFolder, "*", SearchOption.AllDirectories).Select(x => new FileInfo(x))
        .Select(x => new OriginalImageData { FileName = x.Name, FileExtension = x.Extension, Label = x.Directory!.Name, Folder = augmentedDataFolder, Image = Cv2.ImRead(x.FullName) });

    Augmentator.GenerateAugmentedDataset(originalImages);
    Console.WriteLine("Augmentation data generated!");
}

Console.WriteLine("Model Training!");
var (trainedModelPath, testSet) = Trainer.Train(augmentedDataFolder);
Console.WriteLine("Model Trained!");

Console.WriteLine("Model Evaluation!");
Evaluator.Evaluate(trainedModelPath, testSet);
Console.WriteLine("Model Evaluated!");

Console.WriteLine("Bye, World!");
// See https://aka.ms/new-console-template for more information
using BrnoAiDaysWorkshop.Evaluation;
using BrnoAiDaysWorkshop.Training;

Console.WriteLine("Hello, World!");

var (trainedModelPath, testSet) = Trainer.Train("data"); // todo: replace "data" folder with augmented data folder @lukas
Evaluator.Evaluate(trainedModelPath, testSet);

Console.WriteLine("Model trained!");
using Microsoft.ML;
using Microsoft.ML.Data;

namespace BrnoAiDaysWorkshop.Evaluation;

public static class Evaluator {

    public static void Evaluate(FileInfo modelPath, IDataView testSet) {
        var mlContext = new MLContext();

        // todo: TASK 3a Load model from file
        ITransformer trainedModel = null;

        // todo: TASK 3b mass-predict testSet using Transform method on loaded model
        IDataView predictions = null;

        var metrics = mlContext.MulticlassClassification.Evaluate(data: predictions, labelColumnName: "LabelAsKey", scoreColumnName: "Score");
        PrintClassificationMetrics(metrics);
    }

    private static void PrintClassificationMetrics(MulticlassClassificationMetrics metrics) {
        Console.WriteLine("\n\n");
        Console.WriteLine($"************************************************************");
        Console.WriteLine($"*    Evaluation metrics for trained model   ");
        Console.WriteLine($"*-----------------------------------------------------------");
        Console.WriteLine($"    AccuracyMacro = {metrics.MacroAccuracy:0.####}");
        Console.WriteLine($"    AccuracyMicro = {metrics.MicroAccuracy:0.####}");

        Console.WriteLine("\n\n");
        Console.WriteLine("Confusion matrix: \n");
        Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
        Console.WriteLine($"************************************************************");
    }
}
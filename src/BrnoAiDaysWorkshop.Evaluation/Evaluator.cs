using Microsoft.ML;
using Microsoft.ML.Data;

namespace BrnoAiDaysWorkshop.Evaluation {
    public static class Evaluator {

        public static Task Evaluate(FileInfo modelPath, IDataView testSet) {
            var mlContext = new MLContext();

            var trainedModel = mlContext.Model.Load(modelPath.FullName, out _);
            var predictions = trainedModel.Transform(testSet);

            var metrics = mlContext.MulticlassClassification.Evaluate(data: predictions, labelColumnName: "LabelAsKey", scoreColumnName: "Score");
            PrintClassificationMetrics(metrics);

            return Task.CompletedTask;
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
}

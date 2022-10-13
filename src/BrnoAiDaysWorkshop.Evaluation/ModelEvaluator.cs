using Microsoft.ML;
using Microsoft.ML.Data;

namespace BrnoAiDaysWorkshop.Evaluation {
    public static class ModelEvaluator {

        public static Task Evaluate(FileInfo modelPath, IDataView testSet) {
            var mlContext = new MLContext();
            DataViewSchema modelSchema;

            var trainedModel = mlContext.Model.Load(modelPath.FullName, out modelSchema);
            var predictions = trainedModel.Transform(testSet);

            var metrics = mlContext.MulticlassClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
            PrintClassificationMetrics(metrics);

            return Task.CompletedTask;
        }

        private static void PrintClassificationMetrics(MulticlassClassificationMetrics metrics) {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Evaluation metrics for trained model   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"    AccuracyMacro = {metrics.MacroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    AccuracyMicro = {metrics.MicroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 1 = {metrics.PerClassLogLoss[0]:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 2 = {metrics.PerClassLogLoss[1]:0.####}, the closer to 0, the better");
            Console.WriteLine("\n\n");
            Console.WriteLine("Confusion matrix: \n");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
            Console.WriteLine($"************************************************************");
        }
    }
}

using Microsoft.ML.Data;

namespace BrnoAiDaysWorkshop.AutoML.Helpers;

public static class ConsoleHelper
{
    private const int Width = 114;

    public static void PrintMulticlassClassificationMetrics(string name, MulticlassClassificationMetrics metrics)
    {
        Console.WriteLine($"************************************************************");
        Console.WriteLine($"*    Metrics for {name} multi-class classification model    ");
        Console.WriteLine($"*-----------------------------------------------------------");
        Console.WriteLine($"    MacroAccuracy = {metrics.MacroAccuracy:0.####}, a value from 0 and 1, where closer to 1.0 is better");
        Console.WriteLine($"    MicroAccuracy = {metrics.MicroAccuracy:0.####}, a value from 0 and 1, where closer to 1.0 is better");
        Console.WriteLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");
        Console.WriteLine($"    LogLoss for class 1 = {metrics.PerClassLogLoss[0]:0.####}, the closer to 0, the better");
        Console.WriteLine($"    LogLoss for class 2 = {metrics.PerClassLogLoss[1]:0.####}, the closer to 0, the better");
        Console.WriteLine($"    LogLoss for class 3 = {metrics.PerClassLogLoss[2]:0.####}, the closer to 0, the better");
        Console.WriteLine($"************************************************************");
    }

    internal static void PrintIterationMetrics(int iteration, string trainerName, MulticlassClassificationMetrics metrics, double? runtimeInSeconds)
    {
        CreateRow($"{iteration,-4} {trainerName,-35} {metrics?.MicroAccuracy ?? double.NaN,14:F4} {metrics?.MacroAccuracy ?? double.NaN,14:F4} {runtimeInSeconds.Value,9:F1}", Width);
    }

    internal static void PrintIterationException(Exception ex)
    {
        Console.WriteLine($"Exception during AutoML iteration: {ex}");
    }

    internal static void PrintMulticlassClassificationMetricsHeader()
    {
        CreateRow($"{"",-4} {"Trainer",-35} {"MicroAccuracy",14} {"MacroAccuracy",14} {"Duration",9}", Width);
    }

    private static void CreateRow(string message, int width)
    {
        Console.WriteLine("|" + message.PadRight(width - 2) + "|");
    }

    public static void ConsoleWriteHeader(params string[] lines)
    {
        var defaultColor = Console.ForegroundColor;
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine(" ");
        foreach (var line in lines)
        {
            Console.WriteLine(line);
        }
        var maxLength = lines.Select(x => x.Length).Max();
        Console.WriteLine(new string('#', maxLength));
        Console.ForegroundColor = defaultColor;
    }
}
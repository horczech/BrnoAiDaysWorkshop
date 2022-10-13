using Microsoft.ML.Data;

public class PredictionModel
{
    [ColumnName("PredictedLabel")]
    public string PredictedLabel { get; set; }
}
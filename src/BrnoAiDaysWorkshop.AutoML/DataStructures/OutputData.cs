using Microsoft.ML.Data;

namespace BrnoAiDaysWorkshop.AutoML.DataStructures;

internal class OutputData
{
    [ColumnName("Score")]
    public float[] Score;
}
using Microsoft.ML.Data;

namespace BrnoAiDaysWorkshop.Intro.DataStructures
{
    public class OutputData
    {
        [ColumnName("Score")]
        public float[] Score;
    }
}

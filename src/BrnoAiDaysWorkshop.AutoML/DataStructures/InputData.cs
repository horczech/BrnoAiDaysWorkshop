﻿using Microsoft.ML.Data;

namespace BrnoAiDaysWorkshop.AutoML.DataStructures;

internal class InputData
{
    [ColumnName("PixelValues")]
    [VectorType(64)]
    public float[] PixelValues;

    [LoadColumn(64)]
    public float Number;
}
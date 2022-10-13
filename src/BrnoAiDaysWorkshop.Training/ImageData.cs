namespace BrnoAiDaysWorkshop.Training;

public class ImageData
{
    public string ImagePath { get; set; }
    public string Label { get; set; }
    public uint LabelAsKey { get; set; }
    public byte[] Image { get; set; }
}
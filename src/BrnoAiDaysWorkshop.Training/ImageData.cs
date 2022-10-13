namespace BrnoAiDaysWorkshop.Training;

public class ImageData
{
    public string ImagePath { get; set; }
    public string Label { get; set; }
    public uint LabelAsKey { get; set; } //ToDo M: Remove it makes extra useless column on IDataView :(
    public byte[] Image { get; set; }

    public string GetFileName() => new FileInfo(ImagePath).Name;
}
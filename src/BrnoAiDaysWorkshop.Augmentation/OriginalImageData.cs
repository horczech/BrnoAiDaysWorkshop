using OpenCvSharp;
namespace BrnoAiDaysWorkshop.Augmentation;

public class OriginalImageData
{
    public string FileName{ get; set; }
    public string Folder { get; set; }
    public string Label{ get; set; }
    public string FileExtension { get; set; }
    public Mat Image { get; set; }
}
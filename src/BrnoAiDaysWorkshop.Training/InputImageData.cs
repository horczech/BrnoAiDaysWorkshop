namespace BrnoAiDaysWorkshop
{
    public class InputImageData
    {
        public string ImagePath;
        public string Label;

        public string GetFileName() => new FileInfo(ImagePath).Name;
    }
}
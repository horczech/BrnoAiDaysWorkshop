using ImageClassification.DataModels;
using Microsoft.ML;

namespace BrnoAiDaysWorkshop.Training
{
    public static class DatasetLoader
    {
        public static IDataView LoadImages(string imageFolder) {
            var mlContext = new MLContext();

            var files = LoadImagePaths(imageFolder);
            var data = mlContext.Data.LoadFromEnumerable(files);
            var shuffledData = mlContext.Data.ShuffleRows(data);

            return shuffledData;
        }

        private static IEnumerable<ImageData> LoadImagePaths(string pathToDir) {
            var images = new List<ImageData>();
            var files = Directory.GetFiles(pathToDir, "*", SearchOption.AllDirectories).Select(x => new FileInfo(x)).ToList();

            foreach (var file in files) {
                if (file.Extension != ".jpg" && file.Extension != ".png")
                    continue;

                var label = file.Directory!.Name;
                images.Add(new ImageData { ImagePath = $"{file.Directory.Name}\\{file.Name}", Label = label });
            }

            return images;
        }
    }
}
using OpenCvSharp;

namespace BrnoAiDaysWorkshop.Augmentation
{
    public static class Augmentator
    {
        private static readonly Random Random = new Random();

        public static void GenerateAugmentedDataSet(List<OriginalImageData> images) {
            var iterationCounter = 0;
            foreach (var image in images) {
                Console.WriteLine($"Augmentation Progress: {iterationCounter++}/{images.Count}");
                // load image
                using var sourceImage = Cv2.ImRead(image.FilePath);
                // save original image to the data folder
                sourceImage.SaveImage(BuildImagePath(image));

                //todo TASK 4: apply augmentations
                var augmentedImage = sourceImage.Clone();
                //augmentedImage = AugmentationMethods.
                augmentedImage.SaveImage(BuildImagePath(image));
            }
        }

        private static string BuildImagePath(OriginalImageData image) => Path.Combine(image.Folder, image.Label, image.FileName + Random.Next(0, int.MaxValue) + image.FileExtension);
    }
}
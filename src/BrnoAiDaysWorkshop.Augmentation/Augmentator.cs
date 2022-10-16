// #define VISUSAL_DEBUG
using OpenCvSharp;

namespace BrnoAiDaysWorkshop.Augmentation
{
    public static class Augmentator
    {
        private const int NumberOfGenerationsPerOriginalImage = 10;

        public static void GenerateAugmentedDataset(IEnumerable<OriginalImageData> images) {
            var uniformRandom = new Random(42);
            var iterationCounter = 0;
            foreach (var image in images) {
                var augmentedImage = image.Image.Clone();
                Console.WriteLine($"Augmentation Progress: {iterationCounter++}/{images.Count()}");
                image.Image.SaveImage($"{image.Folder}\\{image.Label}\\{image.FileName}-orig{image.FileExtension}");
                for (var i = 0; i < NumberOfGenerationsPerOriginalImage; i++) {
                    image.Image.CopyTo(augmentedImage);
                    var nonZeroMultiplier = 1.0 + i % 5;
#if VISUSAL_DEBUG
                    Cv2.ImShow("orig", augmentedImage);
                    Cv2.ResizeWindow("orig", 200, 200);
#endif
                    augmentedImage = uniformRandom.Next(7) switch {
                        0 => AugmentationMethods.TranslationDeformation(augmentedImage, nonZeroMultiplier * 1, 0),
                        1 => AugmentationMethods.TranslationDeformation(augmentedImage, nonZeroMultiplier * 1, nonZeroMultiplier * 1),
                        2 => AugmentationMethods.TranslationDeformation(augmentedImage, 0, nonZeroMultiplier * 1),
                        3 => AugmentationMethods.TranslationDeformation(augmentedImage, -nonZeroMultiplier * 1, 0),
                        4 => AugmentationMethods.TranslationDeformation(augmentedImage, -nonZeroMultiplier * 1, -nonZeroMultiplier * 1),
                        5 => AugmentationMethods.TranslationDeformation(augmentedImage, 0, -nonZeroMultiplier * 1),
                        _ => augmentedImage,
                    };
#if VISUSAL_DEBUG
                    Cv2.ImShow("translation", augmentedImage);
                    Cv2.ResizeWindow("translation", 200, 200);
#endif
                    augmentedImage = uniformRandom.Next(2) switch {
                        0 => AugmentationMethods.AddNoise(augmentedImage, 0, nonZeroMultiplier * 2),
                        _ => augmentedImage,
                    };
#if VISUSAL_DEBUG
                    Cv2.ImShow("noise", augmentedImage);
                    Cv2.ResizeWindow("noise", 200, 200);
#endif
                    augmentedImage = uniformRandom.Next(3) switch {
                        0 => AugmentationMethods.AdjustContrast(augmentedImage, 1 + nonZeroMultiplier * 0.05, nonZeroMultiplier * -10),
                        1 => AugmentationMethods.AdjustContrast(augmentedImage, 1 - nonZeroMultiplier * 0.05, nonZeroMultiplier * 10),
                        _ => augmentedImage,
                    };
#if VISUSAL_DEBUG
                    Cv2.ImShow("contrast", augmentedImage);
                    Cv2.ResizeWindow("contrast", 200, 200);
#endif
                    augmentedImage = uniformRandom.Next(3) switch {
                        0 => AugmentationMethods.ChangeIntensity(augmentedImage, nonZeroMultiplier * 0.05, false),
                        1 => AugmentationMethods.ChangeIntensity(augmentedImage, nonZeroMultiplier * 0.05, true),
                        _ => augmentedImage,
                    };
#if VISUSAL_DEBUG
                    Cv2.ImShow("intensity", augmentedImage);
                    Cv2.ResizeWindow("intensity", 200, 200);
#endif
                    augmentedImage = uniformRandom.Next(3) switch {
                        0 => AugmentationMethods.ChangeTemperature(augmentedImage, ColormapTypes.Cool, nonZeroMultiplier * 0.01),
                        1 => AugmentationMethods.ChangeTemperature(augmentedImage, ColormapTypes.Hot, nonZeroMultiplier * 0.01),
                        _ => augmentedImage,
                    };
#if VISUSAL_DEBUG
                    Cv2.ImShow("temperature", augmentedImage);
                    Cv2.ResizeWindow("temperature", 200, 200);
#endif
                    augmentedImage = uniformRandom.Next(3) switch {
                        0 => AugmentationMethods.RotateDeformation(augmentedImage, nonZeroMultiplier * 0.5),
                        1 => AugmentationMethods.RotateDeformation(augmentedImage, nonZeroMultiplier * -0.5),
                        _ => augmentedImage,
                    };
#if VISUSAL_DEBUG
                    Cv2.ImShow("rotation", augmentedImage);
                    Cv2.ResizeWindow("rotation", 200, 200);
#endif
                    augmentedImage = uniformRandom.Next(2) switch {
                        0 => AugmentationMethods.ScaleDeformation(augmentedImage, 1 + 0.015 * nonZeroMultiplier),
                        _ => augmentedImage,
                    };
#if VISUSAL_DEBUG
                    Cv2.ImShow("scale", augmentedImage);
                    Cv2.ResizeWindow("scale", 200, 200);
#endif
                    augmentedImage = uniformRandom.Next(2) switch {
                        0 => AugmentationMethods.ApplyGaussFilter(augmentedImage, new Size(3, 3), nonZeroMultiplier*0.5),
                        1 => AugmentationMethods.ApplyGaussFilter(augmentedImage, new Size(5, 5), nonZeroMultiplier*0.5),
                        _ => augmentedImage,
                    };
#if VISUSAL_DEBUG
                    Cv2.ImShow("all augmentation", augmentedImage);
                    Cv2.ResizeWindow("all augmentation", 200, 200);
                    Cv2.WaitKey();
#endif
                    augmentedImage.SaveImage($"{image.Folder}\\{image.Label}\\{image.FileName}-{i}{image.FileExtension}");
                }
            }
        }
    }
}
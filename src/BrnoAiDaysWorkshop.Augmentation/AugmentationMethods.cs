using OpenCvSharp;
using static System.Net.Mime.MediaTypeNames;
using System.Numerics;

namespace BrnoAiDaysWorkshop.Augmentation
{
    public static class AugmentationMethods
    {
        public static Mat AddNoise(Mat source, double mean = 0, double stdDev = 10) {
            var noise = source.Clone();
            Cv2.Randn(noise, Scalar.All(mean), Scalar.All(stdDev));
            Cv2.AddWeighted(source, 1, noise, 1, 0, source);
            return source;
        }

        public static Mat ChangeTemperature(Mat source, ColormapTypes colorMapType, double colorMapRatio) {
            var colorMap = source.Clone();
            Cv2.ApplyColorMap(colorMap, colorMap, colorMapType);
            Cv2.AddWeighted(source, 1 - colorMapRatio, colorMap, colorMapRatio, 0, source);

            return source;
        }

        public static Mat ChangeIntensity(Mat source, double ratio, bool light) {
            if (ratio is > 1.0 or < 0)
                throw new ArgumentOutOfRangeException(nameof(ratio));

            var color = light ? Scalar.White : Scalar.Black;
            var intensityMask = new Mat(source.Size(), source.Type()).SetTo(color);
            Cv2.AddWeighted(source, 1 - ratio, intensityMask, ratio, 0, source);
            return source;
        }

        public static Mat RotateDeformation(Mat source, double angle = 0) {
            var center = new Point2f(source.Width / 2, source.Height / 2);
            var rotationMatrix2D = Cv2.GetRotationMatrix2D(center, angle, 1);
            Cv2.WarpAffine(source, source, rotationMatrix2D, source.Size());
            return source;
        }

        public static Mat ScaleDeformation(Mat source, double scale = 1) {
            var center = new Point2f(source.Width / 2, source.Height / 2);
            var rotationMatrix2D = Cv2.GetRotationMatrix2D(center, 0, scale);
            Cv2.WarpAffine(source, source, rotationMatrix2D, source.Size());
            return source;
        }

        public static Mat TranslationDeformation(Mat source, double offsetX = 0, double offsetY = 0) {
            var doubles = new double[2, 3];
            doubles[0, 0] = 1;
            doubles[0, 1] = 0;
            doubles[0, 2] = offsetX;
            doubles[1, 0] = 0;
            doubles[1, 1] = 1;
            doubles[1, 2] = offsetY;
            var matrix = Mat.FromArray(doubles);
            Cv2.WarpAffine(source, source, matrix, source.Size());
            return source;
        }

        public static Mat ApplyGaussFilter(Mat source, Size kernel, double sigma = 0) => source.GaussianBlur(kernel, sigma);

        public static Mat AdjustContrast(Mat source, double multiplyCoef = 1, double addCoef = 0) {
            Cv2.CvtColor(source, source, ColorConversionCodes.BGR2YCrCb);
            Cv2.Split(source, out var yCrCbChannels);
            Cv2.ConvertScaleAbs(yCrCbChannels[0], yCrCbChannels[0], multiplyCoef, addCoef);
            Cv2.Merge(yCrCbChannels, source);
            Cv2.CvtColor(source, source, ColorConversionCodes.YCrCb2BGR);
            return source;
        }

        public static Mat NormalizeHistogram(Mat source) {
            var histEqualizedImage = source.Clone();
            Cv2.CvtColor(histEqualizedImage, histEqualizedImage, ColorConversionCodes.BGR2YCrCb);
            Cv2.Split(histEqualizedImage, out var vecChannels);
            Cv2.EqualizeHist(vecChannels[0], vecChannels[0]);
            Cv2.Merge(vecChannels, histEqualizedImage);
            Cv2.CvtColor(histEqualizedImage, histEqualizedImage, ColorConversionCodes.YCrCb2BGR);
            return histEqualizedImage;
        }
    }
}
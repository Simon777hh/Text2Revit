using System;

namespace Text2Revit.Addin.Services
{
    public sealed class WindowRuleSampler
    {
        private readonly Random random;

        public WindowRuleSampler(int? seed = null)
        {
            random = seed.HasValue
                ? new Random(seed.Value)
                : new Random();
        }

        public WindowSillAndHeight Sample(string roomType)
        {
            switch ((roomType ?? string.Empty).ToLowerInvariant())
            {
                case "living":
                case "bedroom":
                    return new WindowSillAndHeight(
                        SampleTenth(0.9, 1.0),
                        SampleTenth(1.5, 1.7)
                    );
                case "kitchen":
                    return new WindowSillAndHeight(
                        SampleTenth(0.9, 1.0),
                        SampleTenth(1.2, 1.5)
                    );
                case "bathroom":
                    return new WindowSillAndHeight(
                        SampleTenth(1.2, 1.5),
                        SampleTenth(0.6, 1.2)
                    );
                default:
                    return new WindowSillAndHeight(
                        SampleTenth(0.9, 1.0),
                        SampleTenth(1.2, 1.5)
                    );
            }
        }

        private double SampleTenth(double minimum, double maximum)
        {
            int low = (int)Math.Round(minimum * 10.0);
            int high = (int)Math.Round(maximum * 10.0);
            return random.Next(low, high + 1) / 10.0;
        }
    }

    public readonly struct WindowSillAndHeight
    {
        public WindowSillAndHeight(double sill, double height)
        {
            Sill = sill;
            Height = height;
        }

        public double Sill { get; }
        public double Height { get; }
    }
}

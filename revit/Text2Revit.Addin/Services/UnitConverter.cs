using System;
using Autodesk.Revit.DB;

namespace Text2Revit.Addin.Services
{
    public static class UnitConverter
    {
        public const double MillimetersPerFoot = 304.8;
        public const double MetersPerFoot = 0.3048;

        public static double MillimetersToFeet(double value)
        {
            return value / MillimetersPerFoot;
        }

        public static double MetersToFeet(double value)
        {
            return value / MetersPerFoot;
        }

        public static double PixelsToFeet(
            double value,
            double pixelToMeter)
        {
            return MetersToFeet(value * pixelToMeter);
        }

        public static XYZ PointToXyz(
            double x,
            double y,
            double zMeters,
            double pixelToMeter)
        {
            return new XYZ(
                PixelsToFeet(x, pixelToMeter),
                PixelsToFeet(y, pixelToMeter),
                MetersToFeet(zMeters)
            );
        }
    }
}

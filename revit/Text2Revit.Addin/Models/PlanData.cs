using System.Collections.Generic;
using System.Runtime.Serialization;
namespace Text2Revit.Addin.Models
{
    [DataContract] public sealed class PlanData
    {
        [DataMember(Name="schema_version")] public string SchemaVersion { get; set; }
        [DataMember(Name="coordinate_system")] public string CoordinateSystem { get; set; }
        [DataMember(Name="pixel_to_meter")] public double PixelToMeter { get; set; }
        [DataMember(Name="seed")] public long Seed { get; set; }
        [DataMember(Name="rooms")] public List<RoomData> Rooms { get; set; }
        [DataMember(Name="walls")] public List<WallData> Walls { get; set; }
        [DataMember(Name="railings")] public List<RailingData> Railings { get; set; }
        [DataMember(Name="openings")] public List<OpeningData> Openings { get; set; }
    }
    [DataContract] public sealed class RoomData
    {
        [DataMember(Name="room_id")] public int RoomId { get; set; }
        [DataMember(Name="type")] public string Type { get; set; }
        [DataMember(Name="polygon")] public List<List<double>> Polygon { get; set; }
        [DataMember(Name="interior_point")] public List<double> InteriorPoint { get; set; }
    }
    [DataContract] public sealed class WallData
    {
        [DataMember(Name="wall_id")] public string WallId { get; set; }
        [DataMember(Name="reference_line")] public List<List<double>> ReferenceLine { get; set; }
        [DataMember(Name="host")] public string Host { get; set; }
        [DataMember(Name="owner_room_ids")] public List<int> OwnerRoomIds { get; set; }
        [DataMember(Name="thickness_m")] public double ThicknessM { get; set; }
        [DataMember(Name="height_m")] public double HeightM { get; set; }
    }
    [DataContract] public sealed class RailingData
    {
        [DataMember(Name="railing_id")] public string RailingId { get; set; }
        [DataMember(Name="reference_line")] public List<List<double>> ReferenceLine { get; set; }
        [DataMember(Name="owner_room_id")] public int OwnerRoomId { get; set; }
        [DataMember(Name="height_m")] public double HeightM { get; set; }
    }
    [DataContract] public sealed class OpeningData
    {
        [DataMember(Name="type")] public string Type { get; set; }
        [DataMember(Name="wall_id")] public string WallId { get; set; }
        [DataMember(Name="host")] public string Host { get; set; }
        [DataMember(Name="reference_line")] public List<List<double>> ReferenceLine { get; set; }
        [DataMember(Name="center")] public List<double> Center { get; set; }
        [DataMember(Name="normal")] public List<double> Normal { get; set; }
        [DataMember(Name="width_m")] public double WidthM { get; set; }
        [DataMember(Name="height_m")] public double HeightM { get; set; }
        [DataMember(Name="sill_m")] public double SillM { get; set; }
        [DataMember(Name="owner_room_id")] public int OwnerRoomId { get; set; }
        [DataMember(Name="target_room_id")] public int? TargetRoomId { get; set; }
    }
    [DataContract] internal sealed class BackendRequest
    {
        [DataMember(Name="prompt")] public string Prompt { get; set; }
        [DataMember(Name="language")] public string Language { get; set; }
        [DataMember(Name="seed")] public long? Seed { get; set; }
    }
    [DataContract] internal sealed class BackendStatus
    {
        [DataMember(Name="message")] public string Message { get; set; }
    }
}

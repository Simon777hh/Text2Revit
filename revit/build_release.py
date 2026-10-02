"""Build one offline Windows installer from this repository.
Run Build-Release.cmd, or python build_release.py --python PATH.
All downloads and build products stay inside revit/.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, struct, subprocess, sys, zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
TOOLS = ROOT / "build_tools"
BUILD = ROOT / "build"
DIST = ROOT / "dist"
VERSIONS = {2020: "2020.0.0", 2021: "2021.0.0", 2022: "2022.0.0", 2023: "2023.0.0", 2024: "2024.0.0", 2025: "2025.0.0", 2026: "2026.0.0"}
BACKEND_FILES = ["backend_cli.py", "pipeline_generate.py", "inference_runtime.py", "revit_geometry.py", "plan_quality.py", "healthcheck.py", "data_generation.py", "data_utils.py", "topology_rules.py", "edge_utils.py", "model.py", "flow_solver.py", "runtime_utils.py", "plan_postprocess.py", "restore_canvas_aspect.py", "door_window_rules.py"]
BACKEND_FILES.append("geometry_cleanup.py")
DATA_FILES = ["corner_count_distributions.json", "room_aspect_ranges.json", "room_aspect_samples.npz", "raw_recovery_scale_stats.json", "geometry_policy.json"]

def download(url: str, path: Path):
    if path.is_file() and path.stat().st_size: return
    import requests
    path.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    temporary = path.with_suffix(path.suffix + ".partial")
    print("Downloading", url, flush=True)
    with session.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024): output.write(chunk)
    temporary.replace(path)

def unzip(path: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination.resolve()): raise ValueError("archive path outside destination")
        archive.extractall(destination)

def sdk():
    exe = TOOLS / "dotnet" / "dotnet.exe"
    if exe.is_file(): return exe
    version_file = TOOLS / "sdk-version.txt"
    download("https://builds.dotnet.microsoft.com/dotnet/Sdk/8.0/latest.version", version_file)
    version = version_file.read_text().strip()
    archive = TOOLS / "dotnet-sdk.zip"
    download(f"https://builds.dotnet.microsoft.com/dotnet/Sdk/{version}/dotnet-sdk-{version}-win-x64.zip", archive)
    unzip(archive, exe.parent)
    return exe

def run(command, **kwargs):
    print("Running", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, **kwargs)

def build_plugins(dotnet: Path, years, env):
    outputs = []
    for year in years:
        local = Path(os.environ.get(f"REVIT{year}_API_DIR", str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Autodesk" / f"Revit {year}")))
        if (local / "RevitAPI.dll").is_file(): reference = local
        else:
            version = VERSIONS[year]
            package = "revit_all_main_versions_api_x64"
            archive = TOOLS / "references" / f"{year}.nupkg"
            download(f"https://api.nuget.org/v3-flatcontainer/{package}/{version}/{package}.{version}.nupkg", archive)
            folder = archive.with_suffix("")
            if not folder.exists(): unzip(archive, folder)
            candidates = list(folder.rglob("RevitAPI.dll"))
            if not candidates: raise FileNotFoundError(f"No API reference for Revit {year}")
            reference = candidates[0].parent
        run([dotnet, "build", ROOT / "Text2Revit.Addin/Text2Revit.Addin.csproj", "-c", "Release",
             f"-p:RevitVersion={year}", f"-p:RevitInstallDir={reference}",
             f"-p:BaseIntermediateOutputPath=obj/{year}/", "-v", "minimal"], env=env)
        outputs.append((year, ROOT / f"Text2Revit.Addin/bin/Release/{year}/Text2Revit.Addin.dll"))
    return outputs

def models(python: Path, checkpoint: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    stamp = output / "source.json"
    signature = {"path": str(checkpoint.resolve()), "mtime": checkpoint.stat().st_mtime_ns, "size": checkpoint.stat().st_size}
    if stamp.is_file() and json.loads(stamp.read_text()) == signature and (output / "clip/model.safetensors").is_file(): return
    code = """import sys,torch
from pathlib import Path
from transformers import CLIPTextModel,CLIPTokenizer
source,target=Path(sys.argv[1]),Path(sys.argv[2])
clip_dir=Path(sys.argv[3])
clip_source=str(clip_dir) if clip_dir.is_dir() else 'openai/clip-vit-large-patch14'
checkpoint=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
keys=('hidden','heads','head_dim','node_layers','corner_layers','joint_bit_finetune','base_logits_weight')
args={k:v for k,v in checkpoint['args'].items() if k in keys}
torch.save({'model':checkpoint['model'],'args':args},target/'flow_matching_best.pth')
del checkpoint
model=CLIPTextModel.from_pretrained(clip_source,local_files_only=True)
model.save_pretrained(target/'clip',safe_serialization=True)
CLIPTokenizer.from_pretrained(clip_source,local_files_only=True).save_pretrained(target/'clip')
print('Inference models exported',flush=True)
"""
    run([python, "-c", code, checkpoint, output, PROJECT / "models" / "clip"])
    stamp.write_text(json.dumps(signature), encoding="utf-8")

def runtime(python: Path):
    packed = BUILD / "runtime.zip"
    prefix = python.parent
    stamp = BUILD / "runtime-source.json"
    meta = list((prefix / "conda-meta").glob("*.json"))
    signature = {"prefix": str(prefix), "files": [(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(meta)],
        "pip": [(p.name,p.stat().st_mtime_ns) for p in sorted((prefix/'Lib/site-packages').glob('*.dist-info'))]}
    if packed.exists() and stamp.exists() and json.loads(stamp.read_text()) == json.loads(json.dumps(signature)): return packed
    toolsite = TOOLS / "python"
    if not (toolsite / "conda_pack").is_dir():
        run([python,"-m","pip","install","--target",toolsite,"conda-pack==0.8.1","setuptools<82"])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(toolsite)
    env["PATH"] = str(prefix) + os.pathsep + str(prefix.parent.parent / "Scripts") + os.pathsep + env.get("PATH", "")
    code = "import conda_pack,sys;conda_pack.pack(prefix=sys.argv[1],output=sys.argv[2],format='zip',force=True,compress_level=5,filters=[('exclude','**/__pycache__/*'),('exclude','**/*.pyc'),('exclude','Lib/site-packages/torch/test/*')]);print('Runtime packed',flush=True)"
    run([python,"-c",code,prefix,packed],env=env)
    stamp.write_text(json.dumps(signature),encoding="utf-8")
    return packed

def assemble(stub: Path, plugins, modeldir: Path, packed: Path, version: str):
    archive = BUILD / "payload.zip"
    required = ["runtime/python.exe", "backend/backend_cli.py", "models/flow_matching_best.pth", "models/clip/model.safetensors"]
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as package:
        package.write(packed,"runtime.zip",compress_type=zipfile.ZIP_STORED)
        for filename in BACKEND_FILES: package.write(PROJECT / filename,"backend/" + filename)
        for filename in DATA_FILES:
            source = PROJECT / "data" / filename
            if source.exists():
                if filename == "raw_recovery_scale_stats.json":
                    statistics=json.loads(source.read_text(encoding="utf-8"));statistics.pop("source",None)
                    package.writestr("backend/data/"+filename,json.dumps(statistics,ensure_ascii=False))
                else: package.write(source,"backend/data/" + filename)
            elif filename != "room_aspect_samples.npz": raise FileNotFoundError(source)
        for source in modeldir.rglob("*"):
            if source.is_file() and source.name != "source.json": package.write(source,"models/"+source.relative_to(modeldir).as_posix(),compress_type=zipfile.ZIP_STORED)
        for year,source in plugins:
            name=f"addin/{year}/Text2Revit.Addin.dll";required.append(name);package.write(source,name)
        for filename in ("guide.md","guide.zh.md"):
            package.write(ROOT / filename,filename)
        for filename in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
            package.write(PROJECT / filename, filename)
        for source in (PROJECT / "third_party").glob("*.txt"):
            package.write(source, "third_party/" + source.name)
        with zipfile.ZipFile(packed) as environment:
            peak_bytes=sum(e.file_size for e in environment.infolist())+sum(e.file_size for e in package.infolist())+1024**3
        package.writestr("release.json",json.dumps({"version":version,"revit_versions":[y for y,_ in plugins],"required_files":required,"required_free_bytes":peak_bytes},ensure_ascii=False))
    DIST.mkdir(exist_ok=True)
    final = DIST / "Text2Revit-Setup.exe"
    temporary = final.with_suffix(".partial")
    sha = hashlib.sha256()
    with temporary.open("wb") as output:
        with stub.open("rb") as source: shutil.copyfileobj(source,output)
        offset = output.tell()
        with archive.open("rb") as source:
            while chunk := source.read(1024 * 1024): sha.update(chunk);output.write(chunk)
        length = output.tell()-offset
        output.write(struct.pack("<8sQQ32s",b"T2RPKG01",offset,length,sha.digest()))
    temporary.replace(final)
    run([final,"--verify-only"])
    digest = hashlib.file_digest(final.open("rb"),"sha256").hexdigest()
    (DIST / "SHA256.txt").write_text(digest + "  " + final.name + "\n",encoding="utf-8")
    print(f"READY: {final}\nSize: {final.stat().st_size / 1024**3:.2f} GB",flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python",type=Path,default=Path(sys.executable))
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--years",nargs="+",type=int,default=list(VERSIONS))
    parser.add_argument("--plugins-only",action="store_true")
    args = parser.parse_args()
    if any(y not in VERSIONS for y in args.years): parser.error("Supported years: 2020–2026")
    for folder in (TOOLS,BUILD,DIST): folder.mkdir(exist_ok=True)
    env = os.environ.copy()
    for key in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"): env.pop(key,None)
    env["DOTNET_CLI_HOME"]=str(TOOLS/"cli_home")
    env["NUGET_PACKAGES"]=str(TOOLS/"nuget")
    env["DOTNET_CLI_TELEMETRY_OPTOUT"]="1"
    env["DOTNET_CLI_UI_LANGUAGE"]="en"
    dotnet=sdk()
    plugins=build_plugins(dotnet,args.years,env)
    run([dotnet,"build",ROOT/"Installer/Text2Revit.Installer.csproj","-c","Release","-v","minimal"],env=env)
    if args.plugins_only: return
    candidates=[PROJECT/"checkpoints/flow_matching_best.pth"]
    checkpoint=args.checkpoint or next((p for p in candidates if p.is_file()),None)
    if checkpoint is None: raise FileNotFoundError("Supply --checkpoint with the trained Flow checkpoint")
    modeldir=BUILD/"models"
    models(args.python,checkpoint,modeldir)
    packed=runtime(args.python)
    version="1.0.0-"+datetime.now().strftime("%Y%m%d%H%M%S")
    assemble(ROOT/"Installer/bin/Release/net47/Text2Revit.Setup.exe",plugins,modeldir,packed,version)

if __name__=="__main__": main()

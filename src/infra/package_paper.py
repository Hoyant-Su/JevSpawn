import argparse
from pathlib import Path
import re
import shutil


DEPENDENCIES = re.compile(r"\\(input|include|includegraphics|bibliography)(?:\[[^\]]*\])?\{([^}]+)\}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paper = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.relative_to(paper.parents[2])
    output.mkdir(parents=True, exist_ok=False)
    pending = [paper / "manuscript/main.tex"]
    copied = set()
    while pending:
        source = pending.pop().resolve()
        if source in copied:
            continue
        relative = source.relative_to(paper)
        target = output / (Path(*relative.parts[1:]) if relative.parts[0] == "manuscript" else relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".tex":
            content = source.read_text()
            for command, names in DEPENDENCIES.findall(content):
                for name in names.split(","):
                    dependency = source.parent / name.strip()
                    extension = ".bib" if command == "bibliography" else ".pdf" if command == "includegraphics" else ".tex"
                    pending.append(dependency.with_suffix(extension) if not dependency.suffix else dependency)
            target.write_text(content.replace("../references/", "references/").replace("../results/", "results/"))
        else:
            shutil.copyfile(source, target)
        copied.add(source)
    print(f"Packaged {len(copied)} source files in {output}")


if __name__ == "__main__":
    main()

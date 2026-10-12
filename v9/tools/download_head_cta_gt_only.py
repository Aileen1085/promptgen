"""Download only STL-paired head CTA cases from the pinned Zenodo ZIP.

The caller first downloads the central-directory/footer range, not the CT ZIP.
Downloads resume by compressed byte range and validate every extracted file's CRC.
No files are deleted. Requires requests; does not modify any training environment.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import time
import zipfile
import zlib

import requests

URL = "https://zenodo.org/records/15697196/files/Dataset.zip?download=1"
TOTAL = 25_578_845_008
INDEX_START = 25_567_637_685


class WindowReader(io.RawIOBase):
    """Read a virtual ZIP using its footer and one downloaded physical range."""
    def __init__(self, index: bytes, segment=None, start=0, total=TOTAL,
                 index_start=INDEX_START):
        self.index, self.segment, self.start = index, segment, start
        self.total, self.index_start, self.pos = total, index_start, 0

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (0 if whence == 0 else self.pos if whence == 1 else self.total)
        if self.pos < 0:
            raise ValueError("Negative ZIP position")
        return self.pos

    def read(self, n=-1):
        n = self.total - self.pos if n < 0 else min(n, self.total - self.pos)
        if self.pos >= self.index_start:
            offset = self.pos - self.index_start
            data = self.index[offset:offset + n]
        elif self.segment is not None and self.pos >= self.start:
            self.segment.seek(self.pos - self.start)
            data = self.segment.read(n)
        else:
            raise ValueError("Read outside downloaded ZIP ranges")
        self.pos += len(data)
        return data


def case_id(value):
    value = value.strip()
    return str(int(value)) if value.isdigit() else value.casefold()


def select_files(infos):
    files = [i for i in infos if not i.is_dir()]
    gt = {}
    for info in files:
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
            raise ValueError(f"Unsafe archive member: {info.filename}")
        if path.suffix.casefold() == ".stl":
            m = re.fullmatch(r"an_(\d+)(?:_\d+)?", path.stem, re.IGNORECASE)
            if not m:
                raise ValueError(f"Unrecognized GT case ID: {info.filename}")
            gt.setdefault(case_id(m.group(1)), []).append(info.filename)
    if not gt:
        raise ValueError("Archive contains no recognizable aneurysm GT")
    selected, counts = set(), {k: 0 for k in gt}
    for info in files:
        path = PurePosixPath(info.filename)
        if path.suffix.casefold() in {".stl", ".csv", ".xlsx", ".txt"}:
            selected.add(info.filename)
            continue
        matches = {case_id(p) for p in path.parts[:-1]} & gt.keys()
        if len(matches) > 1:
            raise ValueError(f"Ambiguous CT case path: {info.filename}")
        if matches:
            cid = next(iter(matches))
            selected.add(info.filename)
            counts[cid] += 1
    missing = [k for k, count in counts.items() if not count]
    if missing:
        raise ValueError(f"GT cases without matching CT files: {missing}")
    return selected, gt, counts


def physical_ranges(infos, selected, index_start=INDEX_START, cap=128 * 1024**2):
    """Coalesce adjacent selected members, never include an unselected file body."""
    files = sorted((i for i in infos if not i.is_dir()), key=lambda i: i.header_offset)
    result, current = [], None
    for n, info in enumerate(files):
        end = (files[n + 1].header_offset if n + 1 < len(files) else index_start) - 1
        if info.filename not in selected:
            if current:
                result.append(current)
                current = None
            continue
        if current and end - current["start"] + 1 > cap:
            result.append(current)
            current = None
        if current is None:
            current = {"start": info.header_offset, "end": end, "files": []}
        current["end"] = end
        current["files"].append(info.filename)
    if current:
        result.append(current)
    return result


def fetch_range(url, start, end, path):
    expected = end - start + 1
    for attempt in range(12):
        size = path.stat().st_size if path.exists() else 0
        if size == expected:
            return
        if size > expected:
            raise ValueError(f"Oversized partial file: {path}")
        try:
            with requests.get(url, headers={"Range": f"bytes={start + size}-{end}"},
                              stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                m = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                 response.headers.get("Content-Range", ""))
                if response.status_code != 206 or not m or tuple(map(int, m.groups())) != (start + size, end, TOTAL):
                    raise ValueError("Server returned the wrong byte range")
                with path.open("ab") as output:
                    for data in response.iter_content(1024 * 1024):
                        if output.tell() + len(data) > expected:
                            raise ValueError("Response exceeds requested range")
                        output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
            if path.stat().st_size == expected:
                return
            raise IOError("Incomplete range response")
        except (requests.RequestException, IOError) as error:
            print(f"range={start}-{end} retry={attempt + 1} error={type(error).__name__}", flush=True)
            time.sleep(min(30, 2 * (attempt + 1)))
    raise IOError(f"Could not finish range {start}-{end}")


def crc_file(path):
    value = 0
    with path.open("rb") as source:
        for data in iter(lambda: source.read(1024 * 1024), b""):
            value = zlib.crc32(data, value)
    return value & 0xFFFFFFFF


def extract_range(part, group, lookup, root):
    with part.open("rb") as source:
        for name in group["files"]:
            info = lookup[name]
            target = root.joinpath(*PurePosixPath(name).parts)
            if target.exists():
                if target.stat().st_size != info.file_size or crc_file(target) != info.CRC:
                    raise ValueError(f"Existing output failed CRC; preserved: {target}")
                continue
            if info.flag_bits & 1:
                raise ValueError("Encrypted member")
            source.seek(info.header_offset - group["start"])
            header = source.read(30)
            fields = struct.unpack("<4s5H3L2H", header)
            if fields[0] != b"PK\x03\x04":
                raise ValueError("Invalid local ZIP header")
            source.seek(fields[-2] + fields[-1], 1)
            decoder = zlib.decompressobj(-15) if info.compress_type == 8 else None
            if info.compress_type not in {0, 8}:
                raise ValueError(f"Unsupported compression: {info.compress_type}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".extracting")
            crc, written, remaining = 0, 0, info.compress_size
            with temporary.open("wb") as output:
                while remaining:
                    data = source.read(min(1024 * 1024, remaining))
                    if not data:
                        raise IOError("Truncated compressed member")
                    remaining -= len(data)
                    decoded = decoder.decompress(data) if decoder else data
                    output.write(decoded)
                    crc = zlib.crc32(decoded, crc)
                    written += len(decoded)
                if decoder:
                    decoded = decoder.flush()
                    output.write(decoded)
                    crc = zlib.crc32(decoded, crc)
                    written += len(decoded)
                output.flush()
                os.fsync(output.fileno())
            if written != info.file_size or (crc & 0xFFFFFFFF) != info.CRC:
                raise ValueError(f"CRC/size mismatch: {name}; partial output retained")
            temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        parser.error("workers must be 1..8")
    index_path = args.root / "zip_index.bin"
    if index_path.stat().st_size != TOTAL - INDEX_START:
        raise ValueError("ZIP index download is incomplete; wait for the index job")
    index = index_path.read_bytes()
    with zipfile.ZipFile(WindowReader(index)) as archive:
        infos = archive.infolist()
    selected, gt, counts = select_files(infos)
    if len(gt) != 82:
        raise ValueError(f"Expected 82 GT cases, found {len(gt)}; inspect source before downloading")
    groups = physical_ranges(infos, selected)
    lookup = {i.filename: i for i in infos}
    plan = {"url": URL, "record": 15697196, "total_archive_bytes": TOTAL,
            "gt_case_count": len(gt), "gt_by_case": gt, "ct_files_by_case": counts,
            "selected_files": sorted(selected), "ranges": groups,
            "selected_compressed_bytes": sum(g["end"] - g["start"] + 1 for g in groups)}
    (args.root / "gt_only_download_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"PLAN cases={len(gt)} files={len(selected)} ranges={len(groups)} compressed_bytes={plan['selected_compressed_bytes']}", flush=True)
    if args.plan_only:
        return
    parts = args.root / ".parts"
    parts.mkdir(exist_ok=True)
    def task(group):
        part = parts / f"{group['start']}_{group['end']}.part"
        fetch_range(URL, group["start"], group["end"], part)
        extract_range(part, group, lookup, args.root)
        return len(group["files"])
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        pending = [pool.submit(task, group) for group in groups]
        done = 0
        for future in concurrent.futures.as_completed(pending):
            done += future.result()
            print(f"PROGRESS verified_files={done}/{len(selected)}", flush=True)
    print(f"COMPLETE verified_files={len(selected)} GT_cases={len(gt)} excluded_controls=90", flush=True)


if __name__ == "__main__":
    main()

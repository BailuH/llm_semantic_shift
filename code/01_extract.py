import argparse
import os
import subprocess
import sys
from pathlib import Path

from utils import SEED

WIKIEXTRACTOR = str(Path(sys.executable).parent / "wikiextractor")

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")

PERIOD_MAP = {
    "2021": "zhwiki-20210601-pages-articles-multistream.xml.bz2",
    "2022": "zhwiki-20220501-pages-articles-multistream.xml.bz2",
    "2026": "zhwiki-20260601-pages-articles-multistream.xml.bz2",
}


def extract_period(period, fname):
    bz2_path = RAW_DIR / fname
    if not bz2_path.exists():
        print(f"[跳过] 未找到文件: {bz2_path}")
        return

    out_dir = INTERIM_DIR / f"raw_{period}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        WIKIEXTRACTOR, str(bz2_path),
        "--json",
        "--no-templates",
        "--processes", str(os.cpu_count()),
        "-o", str(out_dir),
    ]

    print(f"  输入: {bz2_path} ({bz2_path.stat().st_size / 1e9:.1f} GB)")
    print(f"  输出: {out_dir}/")
    print(f"  执行: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"  [完成] {period} → {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="WikiExtractor 提取中文维基百科纯文本")
    parser.add_argument(
        "period", nargs="?", default="all",
        choices=["2021", "2022", "2026", "all"],
        help="时段标签，默认 all"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("第 1 步：WikiExtractor 纯文本提取")
    print("=" * 60)

    for period, fname in PERIOD_MAP.items():
        if args.period != "all" and period != args.period:
            continue
        print(f"\n处理 {period} 时段 ...")
        extract_period(period, fname)

    print("\n完成。")


if __name__ == "__main__":
    main()

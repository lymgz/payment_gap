"""Build one SQLite database containing the legacy source data and glossary."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyreadstat

try:
    from src.data_loader import INPUT_FILES
    from src.reporting import sha256_file
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data_loader import INPUT_FILES
    from src.reporting import sha256_file


TABLE_COLUMNS = [
    "code",
    "color",
    "y",
    "x1",
    "x2",
    "x3",
    "x4",
    "x5",
    "x6",
    "x7",
    "Overduearea",
    "Planneddeliveryarea",
    "Province",
    "City",
    "Region",
    "province_additional",
    "city_additional",
    "region_additional",
    "firm_id",
    "completion_pct",
    "months_overdue",
    "offset_discount",
    "gov_intervention",
    "contractor_litigation",
    "contractor_soe",
    "x_coord",
    "y_coord",
    "id",
    "name",
    "ename",
    "Red_O",
    "Yellow_O",
    "Green_O",
    "_ID",
    "_X",
    "_Y",
]

TEXT_COLUMNS = {
    "code",
    "Province",
    "City",
    "Region",
    "province_additional",
    "city_additional",
    "region_additional",
    "name",
    "ename",
}

SOURCE_COLUMN_MAP = {
    "province_additional": "province",
    "city_additional": "city",
    "region_additional": "region",
}

SUPPLEMENTAL_FILE = "supplement_template.xlsx"
SUPPLEMENTAL_SHEET = "supplement-1"
SUPPLEMENTAL_COLUMNS = [
    "code",
    "province",
    "city",
    "region",
    "color",
    "firm_id",
    "completion_pct",
    "months_overdue",
    "offset_discount",
    "gov_intervention",
    "contractor_litigation",
    "contractor_soe",
]


GLOSSARY_ROWS = [
    ("code", "项目代码", "主分析数据中的项目或样本代码。", "data.dta", "code", "文本", "项目标识", "保留原始值，不作为当前主模型的解释变量。"),
    ("color", "风险颜色分档", "项目风险等级的数值编码。", "data.dta", "color", "1=Red；2=Yellow；3=Green", "分类变量", "用于颜色分组、描述统计、回归和倾向得分模型。"),
    ("y", "逾期交付面积", "项目逾期交付面积，是旧代码中的核心因变量。", "data.dta", "y", "面积，单位沿用原始数据", "连续变量", "主模型使用；需与 Overduearea 核对。"),
    ("x1", "预售监控账户余额", "项目预售监控账户余额。", "data.dta", "x1", "连续值，单位沿用原始数据", "连续变量", "回归和倾向得分模型协变量。"),
    ("x2", "商票跳票指标", "项目是否发生商票跳票的二元处理变量。", "data.dta", "x2", "0=否；1=是", "二元变量", "用于分组、回归和倾向得分匹配。"),
    ("x3", "新增产值支付比例（C值）", "新增产值相关的支付比例。", "data.dta", "x3", "比例", "连续变量", "具体分子、分母和统计期间沿用原始研究定义。"),
    ("x4", "实际支付与未支付比例（R值）", "实际支付与未支付相关的比例指标。", "data.dta", "x4", "比例", "连续变量", "具体分子、分母和统计期间沿用原始研究定义。"),
    ("x5", "实际付现/计划付现", "实际付现额与计划付现额之比。", "data.dta", "x5", "比例", "连续变量", "回归和倾向得分模型协变量。"),
    ("x6", "结算款付现/计划结算款", "结算款实际付现额与计划结算款之比。", "data.dta", "x6", "比例", "连续变量", "回归和倾向得分模型协变量。"),
    ("x7", "工抵房价值/计划付款", "工抵房价值与计划付款之比。", "data.dta", "x7", "比例", "连续变量", "回归和倾向得分模型协变量。"),
    ("Overduearea", "逾期交付面积（原始字段）", "原始主数据中的逾期交付面积字段。", "data.dta; province_map.dta", "Overduearea", "面积，单位沿用原始数据", "连续变量", "分析代码派生 Overdue。"),
    ("Planneddeliveryarea", "计划交付面积（原始字段）", "原始主数据中的计划交付面积字段。", "data.dta; province_map.dta", "Planneddeliveryarea", "面积，单位沿用原始数据", "连续变量", "分析代码派生 Planned，并作为 O_rate 分母。"),
    ("Province", "省份", "项目所在省份。", "data.dta", "Province", "文本", "分组变量", "用于省份频数和地图关联。"),
    ("City", "城市", "项目所在城市。", "data.dta", "City", "文本", "地理变量", "保留原始值，当前主模型未直接使用。"),
    ("Region", "区域", "项目所属区域。", "data.dta", "Region", "Central area/Western area/Eastern area", "分组变量", "用于聚类、分组和 MixedLM。"),
    ("x_coord", "横坐标", "项目或省级地图标签的横坐标。", "data.dta; province_map.dta; china_label.dta", "x_coord", "地图坐标", "地理变量", "坐标体系沿用原始 Stata 地图文件。"),
    ("y_coord", "纵坐标", "项目或省级地图标签的纵坐标。", "data.dta; province_map.dta; china_label.dta", "y_coord", "地图坐标", "地理变量", "坐标体系沿用原始 Stata 地图文件。"),
    ("id", "区域编号", "省级区域编号，用于地图标签或省级汇总数据。", "province_map.dta; china_label.dta", "id", "整数编号", "地图关联键", "与 china_map 的 _ID 用于地图几何关联。"),
    ("name", "区域中文名称", "省级区域中文名称。", "province_map.dta; china_label.dta", "name", "文本", "地图标签", "保存原始字段。"),
    ("ename", "区域英文名称", "省级区域英文名称。", "province_map.dta; china_label.dta", "ename", "文本", "地图标签", "保存原始字段。"),
    ("Red_O", "红色项目逾期面积", "省级汇总中红色风险项目的逾期面积。", "province_map.dta", "Red_O", "面积，单位沿用原始数据", "省级汇总", "用于省级风险颜色地图。"),
    ("Yellow_O", "黄色项目逾期面积", "省级汇总中黄色风险项目的逾期面积。", "province_map.dta", "Yellow_O", "面积，单位沿用原始数据", "省级汇总", "用于省级风险颜色地图。"),
    ("Green_O", "绿色项目逾期面积", "省级汇总中绿色风险项目的逾期面积。", "province_map.dta", "Green_O", "面积，单位沿用原始数据", "省级汇总", "用于省级风险颜色地图。"),
    ("_ID", "地图区域编号", "地图多边形对应的区域编号。", "china_map.dta", "_ID", "整数编号", "地图几何键", "通常与省级地图汇总中的 id 对应。"),
    ("_X", "地图多边形横坐标", "地图多边形顶点的横坐标。", "china_map.dta", "_X", "地图坐标", "地图几何", "缺失值可表示多边形之间的分隔行。"),
    ("_Y", "地图多边形纵坐标", "地图多边形顶点的纵坐标。", "china_map.dta", "_Y", "地图坐标", "地图几何", "缺失值可表示多边形之间的分隔行。"),
    ("Overdue", "逾期交付面积（分析字段）", "由 Overduearea 复制生成的分析字段。", "analysis", "Overduearea", "面积，单位沿用原始数据", "派生变量", "不改变原始数据，仅供分析输出使用。"),
    ("Planned", "计划交付面积（分析字段）", "由 Planneddeliveryarea 复制生成的分析字段。", "analysis", "Planneddeliveryarea", "面积，单位沿用原始数据", "派生变量", "不改变原始数据，仅供分析输出使用。"),
    ("O_rate", "逾期率", "逾期交付面积除以计划交付面积。", "analysis", "Overdue / Planned", "比例", "派生变量", "当 Planned 小于等于 0 时设为缺失。"),
    ("color_text", "风险颜色文本", "将 color 的数字编码映射为 Red、Yellow、Green。", "analysis", "color", "Red/Yellow/Green", "派生变量", "用于图形、分组统计和模型分类变量。"),
    ("Region_n", "区域数字编码", "将 Region 文本映射为区域数字编码。", "analysis", "Region", "Central=1；Western=2；Eastern=3", "派生变量", "用于与旧代码保持一致的区域编码。"),
    ("province_additional", "省份（增补字段）", "增补数据中的项目所在省份。", "supplemental", "province", "文本", "增补变量", "数据库字段名为 province_additional，与旧主数据的 Province 字段并列保存，不覆盖原字段。"),
    ("city_additional", "城市（增补字段）", "增补数据中的项目所在城市。", "supplemental", "city", "文本", "增补变量", "数据库字段名为 city_additional，与旧主数据的 City 字段并列保存，不覆盖原字段。"),
    ("region_additional", "区域（增补字段）", "增补数据中的项目所属区域。", "supplemental", "region", "Central area/Western area/Eastern area", "增补变量", "数据库字段名为 region_additional，与旧主数据的 Region 字段并列保存，不覆盖原字段。"),
    ("firm_id", "项目当前操盘方代码", "增补表中的项目当前操盘方或处置阶段代码。", "supplemental", "firm_id", "0=原开发商；1=代建方；2=地方国企/城投；3=司法重整或破产", "增补变量", "实际工作簿使用 0–3 数值编码；与模板中 F01–F10 的文字说明存在口径差异，暂按原值保存。"),
    ("completion_pct", "工程形象进度字段", "增补表中的工程形象进度字段。", "supplemental", "completion_pct", "工作簿实际为 1–5；模板文字写为 0–100", "增补变量", "工作簿 codebook 将其解释为施工节点代码，并非连续百分比；在定义确认前不重编码。"),
    ("months_overdue", "逾期时间分档代码", "增补表中的逾期时间分档字段。", "supplemental", "months_overdue", "1=6个月及以内；2=7–12个月；3=13–24个月；4=24个月以上", "增补变量", "实际工作簿使用 1–4 分档代码，不是实际月数。"),
    ("offset_discount", "抵房折扣", "抵账价除以同类房源市场价（挂牌价）。", "supplemental", "offset_discount", "比例，例如 0.85 表示 8.5 折", "增补变量", "只有部分项目可能适用；缺失值保留为空。"),
    ("gov_intervention", "政府介入程度", "政府对项目风险处置的介入程度。", "supplemental", "gov_intervention", "0=无；1=行政协调；2=定向放款；3=财政注资", "增补变量", "仅按工作簿实际编码保存。"),
    ("contractor_litigation", "承包商起诉", "调查时点前是否发生承包商诉讼、优先受偿申请或正式停工。", "supplemental", "contractor_litigation", "0=否；1=是", "增补变量", "二元指标。"),
    ("contractor_soe", "承包商国企属性", "项目总包单位是否为国企或央企。", "supplemental", "contractor_soe", "0=民企；1=国企/央企", "增补变量", "二元指标。"),
    ("dataset_name", "数据集名称", "all_data 表中记录所属的标准数据集名称。", "database", "dataset_name", "main/province_map/china_map/china_label", "数据库元字段", "用于区分四个原始 Stata 文件。"),
    ("source_file", "来源文件", "all_data 表中每条记录对应的原始文件名。", "database", "source_file", "文件名", "数据库元字段", "用于追溯记录来源。"),
    ("source_row", "来源行号", "记录在原始数据文件中的行号，从 1 开始。", "database", "source_row", "正整数", "数据库元字段", "用于定位原始记录。"),
    ("source_sha256", "来源文件哈希", "原始副本文件的 SHA-256 校验值。", "database", "source_sha256", "64 位十六进制字符串", "数据库元字段", "用于确认数据库数据来自哪个文件版本。"),
    ("row_json", "原始行 JSON", "按原始字段名保存的完整行记录。", "database", "row_json", "JSON 文本", "数据库元字段", "用于保留异构数据集的全部字段和原始字段名。"),
]


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy scalars and missing values to JSON-safe values."""

    if value is None or value is pd.NA:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _sqlite_value(value: Any, column: str) -> Any:
    value = _json_safe(value)
    if value is None:
        return None
    if column in TEXT_COLUMNS:
        return str(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _load_one(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Stata input: {path}")
    frame, metadata = pyreadstat.read_dta(path, apply_value_formats=False)
    return frame, {
        "filename": path.name,
        "sha256": sha256_file(path),
        "row_count": int(len(frame)),
        "column_labels": metadata.column_names_to_labels,
    }


def _load_supplemental(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read the populated supplemental worksheet and exclude fully blank rows."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing supplemental input: {path}")
    frame = pd.read_excel(path, sheet_name=SUPPLEMENTAL_SHEET)
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = [column for column in SUPPLEMENTAL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Supplemental sheet is missing required columns: {missing}")
    frame = frame.dropna(how="all").reset_index(drop=True)
    if frame["code"].isna().any():
        raise ValueError("Supplemental sheet contains rows without code")
    return frame, {
        "filename": path.name,
        "sha256": sha256_file(path),
        "row_count": int(len(frame)),
        "sheet_name": SUPPLEMENTAL_SHEET,
        "columns": [str(column) for column in frame.columns],
    }


def _validate_unique_project_codes(dataset_name: str, frame: pd.DataFrame) -> None:
    """Reject missing or duplicate project keys before creating the database."""

    if "code" not in frame.columns:
        raise ValueError(f"{dataset_name} is missing required project key: code")
    normalized = frame["code"].astype("string").str.strip()
    if normalized.isna().any() or normalized.eq("").any():
        raise ValueError(f"{dataset_name} contains missing code values")
    duplicated = normalized[normalized.duplicated(keep=False)].drop_duplicates().tolist()
    if duplicated:
        preview = duplicated[:10]
        raise ValueError(f"{dataset_name} contains duplicate code values: {preview}")


def _create_schema(connection: sqlite3.Connection) -> None:
    columns = [
        '"record_id" INTEGER PRIMARY KEY',
        '"dataset_name" TEXT NOT NULL',
        '"source_file" TEXT NOT NULL',
        '"source_row" INTEGER NOT NULL',
        '"source_sha256" TEXT NOT NULL',
    ]
    for column in TABLE_COLUMNS:
        sql_type = "TEXT" if column in TEXT_COLUMNS else "REAL"
        columns.append(f'"{column}" {sql_type}')
    columns.append('"row_json" TEXT NOT NULL')
    connection.execute(f'CREATE TABLE "all_data" ({", ".join(columns)})')
    connection.execute('CREATE INDEX "idx_all_data_dataset" ON "all_data" ("dataset_name")')
    connection.execute('CREATE INDEX "idx_all_data_source" ON "all_data" ("source_file")')

    connection.execute(
        """
        CREATE TABLE "glossary" (
            "glossary_id" INTEGER PRIMARY KEY,
            "term_key" TEXT NOT NULL,
            "term_name_cn" TEXT NOT NULL,
            "definition_cn" TEXT NOT NULL,
            "applicable_dataset" TEXT NOT NULL,
            "source_variable" TEXT NOT NULL,
            "data_type_or_coding" TEXT NOT NULL,
            "category" TEXT NOT NULL,
            "notes" TEXT
        )
        """
    )
    connection.execute('CREATE UNIQUE INDEX "idx_glossary_term_dataset" ON "glossary" ("term_key", "applicable_dataset")')
    connection.execute(
        """
        CREATE VIEW "project_analysis" AS
        SELECT
            m."code",
            m."color",
            m."y",
            m."x1",
            m."x2",
            m."x3",
            m."x4",
            m."x5",
            m."x6",
            m."x7",
            m."Overduearea",
            m."Planneddeliveryarea",
            m."Province",
            m."City",
            m."Region",
            s."province_additional",
            s."city_additional",
            s."region_additional",
            s."firm_id",
            s."completion_pct",
            s."months_overdue",
            s."offset_discount",
            s."gov_intervention",
            s."contractor_litigation",
            s."contractor_soe",
            m."source_file" AS "main_source_file",
            m."source_sha256" AS "main_source_sha256",
            s."source_file" AS "supplemental_source_file",
            s."source_sha256" AS "supplemental_source_sha256",
            m."row_json" AS "main_row_json",
            s."row_json" AS "supplemental_row_json"
        FROM "all_data" AS m
        LEFT JOIN "all_data" AS s
          ON s."dataset_name" = 'supplemental'
         AND s."code" = m."code"
        WHERE m."dataset_name" = 'main'
        """
    )


def build_database(
    input_dir: Path,
    output_db: Path,
    additional_dir: Path | None = None,
) -> dict[str, Any]:
    """Read Stata inputs and, when supplied, the supplemental Excel sheet into one database."""

    input_dir = Path(input_dir).resolve()
    output_db = Path(output_db).resolve()
    output_db.parent.mkdir(parents=True, exist_ok=True)

    loaded: list[tuple[str, Path, pd.DataFrame, dict[str, Any]]] = []
    for dataset_name, filename in INPUT_FILES.items():
        path = input_dir / filename
        frame, metadata = _load_one(path)
        loaded.append((dataset_name, path, frame, metadata))

    supplemental_path: Path | None = None
    if additional_dir is not None:
        additional_dir = Path(additional_dir).resolve()
        supplemental_path = additional_dir / SUPPLEMENTAL_FILE
        if not supplemental_path.is_file():
            raise FileNotFoundError(f"Supplemental workbook not found: {supplemental_path}")
        frame, metadata = _load_supplemental(supplemental_path)
        loaded.append(("supplemental", supplemental_path, frame, metadata))

    for dataset_name, _, frame, _ in loaded:
        if dataset_name in {"main", "supplemental"}:
            _validate_unique_project_codes(dataset_name, frame)

    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{output_db.stem}-",
        suffix=".tmp",
        dir=output_db.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    connection = sqlite3.connect(temporary_path)
    try:
        connection.execute("PRAGMA encoding = 'UTF-8'")
        _create_schema(connection)
        insert_columns = [
            "record_id",
            "dataset_name",
            "source_file",
            "source_row",
            "source_sha256",
            *TABLE_COLUMNS,
            "row_json",
        ]
        quoted_columns = ", ".join(f'"{column}"' for column in insert_columns)
        placeholders = ", ".join("?" for _ in insert_columns)
        insert_sql = f'INSERT INTO "all_data" ({quoted_columns}) VALUES ({placeholders})'
        record_id = 1
        counts: dict[str, int] = {}
        for dataset_name, path, frame, metadata in loaded:
            count = 0
            source_row_start = 2 if dataset_name == "supplemental" else 1
            for source_row, (_, row) in enumerate(frame.iterrows(), start=source_row_start):
                raw = {str(column): _json_safe(row[column]) for column in frame.columns}
                values: list[Any] = [record_id, dataset_name, path.name, source_row, metadata["sha256"]]
                values.extend(
                    _sqlite_value(row[source_column], source_column)
                    if (source_column := SOURCE_COLUMN_MAP.get(column, column)) in frame.columns
                    else None
                    for column in TABLE_COLUMNS
                )
                values.append(json.dumps(raw, ensure_ascii=False, separators=(",", ":")))
                connection.execute(insert_sql, values)
                record_id += 1
                count += 1
            counts[dataset_name] = count

        connection.executemany(
            """
            INSERT INTO "glossary" (
                "term_key", "term_name_cn", "definition_cn", "applicable_dataset",
                "source_variable", "data_type_or_coding", "category", "notes"
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            GLOSSARY_ROWS,
        )
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if not integrity or integrity[0] != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")
        main_count = counts.get("main", 0)
        project_view_count = connection.execute('SELECT COUNT(*) FROM "project_analysis"').fetchone()[0]
        if project_view_count != main_count:
            raise RuntimeError(
                f"project_analysis row count {project_view_count} does not match main row count {main_count}"
            )
        result = {
            "database": str(output_db),
            "input_dir": str(input_dir),
            "table_count": 2,
            "data_row_count": record_id - 1,
            "glossary_row_count": len(GLOSSARY_ROWS),
            "dataset_row_counts": counts,
            "source_hashes": {metadata["filename"]: metadata["sha256"] for _, _, _, metadata in loaded},
            "supplemental_file": str(supplemental_path) if supplemental_path is not None else None,
        }
    except Exception:
        connection.close()
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        connection.close()
    try:
        os.replace(temporary_path, output_db)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return result


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root / "data" / "legacy",
        help="Directory containing the four .dta input files.",
    )
    parser.add_argument(
        "--output-db",
        type=Path,
        default=project_root / "data" / "current" / "disorder_payment_data.db",
        help="Path for the single SQLite database file.",
    )
    parser.add_argument(
        "--additional-dir",
        type=Path,
        default=project_root / "data" / "additional_data",
        help="Directory containing supplement_template.xlsx; pass an empty path only when no supplement is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_database(args.input_dir, args.output_db, args.additional_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

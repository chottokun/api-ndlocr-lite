import sys
import os
import threading
import numpy as np
from PIL import Image
from defusedxml import ElementTree as ET
# defusedxmlはXMLパース専用でSubElementを提供しないため、
# XML要素の生成には標準ライブラリを使用する（セキュリティリスクなし）
from xml.etree.ElementTree import SubElement
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from yaml import safe_load

# Add submodule src to path to allow imports from it
# This is necessary because the ndlocr-lite submodule expects its internal structure to be on the PYTHONPATH.
SUBMODULE_SRC = Path(__file__).resolve().parent.parent.parent / "extern" / "ndlocr-lite" / "src"
if str(SUBMODULE_SRC) not in sys.path:
    sys.path.append(str(SUBMODULE_SRC))

from deim import DEIM  # noqa: E402
from parseq import PARSEQ  # noqa: E402
from reading_order.xy_cut.eval import eval_xml  # noqa: E402
from ndl_parser import convert_to_xml_string3  # noqa: E402
from character_inclusion import apply_character_inclusion  # noqa: E402
from ocr import build_markdown_text_from_page, _normalize_markdown_styles  # noqa: E402
from tablecenternet import (  # noqa: E402
    TableCenterNetRecognizer,
    cells_to_markdown,
    infer_tables_from_page,
)

class RecogLine:
    """
    Data class representing a line image and its metadata for recognition.
    Supports comparison for sorting based on original detection index.
    """
    def __init__(self, npimg: np.ndarray, idx: int, pred_char_cnt: int, pred_str: str = ""):
        self.npimg = npimg
        self.idx = idx
        self.pred_char_cnt = pred_char_cnt
        self.pred_str = pred_str

    def __lt__(self, other):
        return self.idx < other.idx

class NDLOCREngine:
    """
    Wrapper for the NDLOCR-Lite engine.
    Handles layout detection, reading order analysis, and character recognition
    using a multi-model cascading strategy.
    """

    # Constants for cascade logic (thresholds for routing to different models)
    CASCADE_PRED_CHAR_SMALL = 3.0
    CASCADE_PRED_CHAR_MEDIUM = 2.0
    CASCADE_RECOG30_MAX_LEN = 25
    CASCADE_RECOG50_MAX_LEN = 45

    def __init__(
        self,
        device: str = "cpu",
        det_weights: Optional[str] = None,
        det_classes: Optional[str] = None,
        rec_weights: Optional[str] = None,
        rec_weights30: Optional[str] = None,
        rec_weights50: Optional[str] = None,
        rec_classes: Optional[str] = None,
        det_score_threshold: float = 0.2,
        det_conf_threshold: float = 0.25,
        det_iou_threshold: float = 0.2,
        enable_tcy: bool = False,
        enable_character_inclusion: bool = False,
        enable_rich_markdown: bool = False,
        enable_table_structure: bool = False,
        table_weights: Optional[str] = None,
    ):
        """
        Initializes the engine with model paths and detection thresholds.
        Loads ONNX models into memory.
        """
        self.device = device
        self.enable_tcy = enable_tcy
        self.enable_character_inclusion = enable_character_inclusion
        self.enable_rich_markdown = enable_rich_markdown
        self.enable_table_structure = enable_table_structure
        
        # ndlocr-lite 2.0.0 (v2) のデフォルトモデルパス
        base_dir = SUBMODULE_SRC
        self.det_weights = det_weights or str(base_dir / "model" / "deim-s-1024x1024-261124.onnx")
        self.det_classes = det_classes or str(base_dir / "config" / "ndl.yaml")
        self.rec_weights = rec_weights or str(base_dir / "model" / "parseq-ndl-24x768-100-tiny-153epoch-261124.onnx")
        self.rec_weights30 = rec_weights30 or str(base_dir / "model" / "parseq-ndl-24x256-30-tiny-499epoch-261124.onnx")
        self.rec_weights50 = rec_weights50 or str(base_dir / "model" / "parseq-ndl-24x384-50-tiny-278epoch-261124.onnx")
        self.rec_classes = rec_classes or str(base_dir / "config" / "NDLmoji.yaml")
        self.table_weights = table_weights or str(base_dir / "model" / "tablecenternet_s_800_v3.onnx")
        
        self.det_score_threshold = det_score_threshold
        self.det_conf_threshold = det_conf_threshold
        self.det_iou_threshold = det_iou_threshold

        self.detector = None
        self.recognizer100 = None
        self.recognizer30 = None
        self.recognizer50 = None
        self.table_recognizer = None
        self._table_recognizer_lock = threading.Lock()
        
        # ThreadPoolExecutor for parallelizing character recognition across lines
        # Allow override via environment variable OCR_MAX_WORKERS, or use a bounded default.
        # Too many concurrent threads calling ORT sessions on CPU causes high contention and thrashing.
        max_workers_env = os.getenv("OCR_MAX_WORKERS")
        cpu_count = os.cpu_count() or 4
        default_workers = min(4, cpu_count)
        if max_workers_env:
            try:
                max_workers = max(1, int(max_workers_env))
            except ValueError:
                max_workers = default_workers
        else:
            max_workers = default_workers

        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ocr_worker"
        )

        self._load_models()

    def _load_models(self):
        """Loads ONNX models for detection and recognition."""
        print(f"[INFO] Loading detector from {self.det_weights}")
        self.detector = DEIM(
            model_path=self.det_weights,
            class_mapping_path=self.det_classes,
            score_threshold=self.det_score_threshold,
            conf_threshold=self.det_conf_threshold,
            iou_threshold=self.det_iou_threshold,
            device=self.device
        )

        print("[INFO] Loading recognizers")
        self.recognizer100 = self._get_recognizer(self.rec_weights)
        self.recognizer30 = self._get_recognizer(self.rec_weights30)
        self.recognizer50 = self._get_recognizer(self.rec_weights50)

        if self.enable_table_structure:
            self._ensure_table_recognizer()

    def _ensure_table_recognizer(self):
        """表構造化モデルをスレッドセーフに遅延ロードする。"""
        if self.table_recognizer is not None:
            return self.table_recognizer
        with self._table_recognizer_lock:
            if self.table_recognizer is None:
                print(f"[INFO] Loading table recognizer from {self.table_weights}")
                self.table_recognizer = TableCenterNetRecognizer(
                    model_path=self.table_weights,
                    device=self.device,
                )
        return self.table_recognizer

    def _get_recognizer(self, weights_path: str):
        """Helper to initialize a PARSEQ recognizer with a specific weight file."""
        with open(self.rec_classes, encoding="utf-8") as f:
            charobj = safe_load(f)
        charlist = list(charobj["model"]["charset_train"])
        recognizer = PARSEQ(model_path=weights_path, charlist=charlist, device=self.device)
        
        if self.enable_tcy:
            try:
                from tcy_wrapper import TateChuYokoWrapper
                recognizer = TateChuYokoWrapper(recognizer)
                print(f"[INFO] TCY enabled for recognizer: {weights_path}")
            except ImportError:
                print("[WARNING] Could not import TateChuYokoWrapper. TCY will be disabled.")
        
        return recognizer

    def shutdown(self):
        """Shuts down the internal thread pool."""
        self.executor.shutdown(wait=False, cancel_futures=True)

    def _process_cascade(self, alllineobj: List[RecogLine], is_cascade: bool = True) -> List[str]:
        """
        Recognition cascading strategy.
        1. Routes lines to PARSEQ-30, 50, or 100 based on 'pred_char_cnt' from the detector.
        2. If a smaller model yields a result longer than its training length, it cascades to the next larger model.
        3. For extremely long lines (>=98 chars), splits the line and re-recognizes (v1.2.1 improvement).
        """
        targetdflist30 = []
        targetdflist50 = []
        targetdflist100 = []
        targetdflist200 = [] # For split lines
        
        for lineobj in alllineobj:
            if lineobj.pred_char_cnt == self.CASCADE_PRED_CHAR_SMALL and is_cascade:
                targetdflist30.append(lineobj)
            elif lineobj.pred_char_cnt == self.CASCADE_PRED_CHAR_MEDIUM and is_cascade:
                targetdflist50.append(lineobj)
            else:
                targetdflist100.append(lineobj)
        
        targetdflistall = []

        # Level 1: PARSEQ-30 (Fastest, short lines)
        if len(targetdflist30) > 0:
            resultlines30 = list(self.executor.map(self.recognizer30.read, [t.npimg for t in targetdflist30]))
            for i, pred_str in enumerate(resultlines30):
                lineobj = targetdflist30[i]
                if len(pred_str) >= self.CASCADE_RECOG30_MAX_LEN:
                    targetdflist50.append(lineobj) # Cascade up
                else:
                    lineobj.pred_str = pred_str
                    targetdflistall.append(lineobj)

        # Level 2: PARSEQ-50 (Medium lines)
        if len(targetdflist50) > 0:
            resultlines50 = list(self.executor.map(self.recognizer50.read, [t.npimg for t in targetdflist50]))
            for i, pred_str in enumerate(resultlines50):
                lineobj = targetdflist50[i]
                if len(pred_str) >= self.CASCADE_RECOG50_MAX_LEN:
                    targetdflist100.append(lineobj) # Cascade up
                else:
                    lineobj.pred_str = pred_str
                    targetdflistall.append(lineobj)

        # Level 3: PARSEQ-100 (Highest capacity, long lines)
        if len(targetdflist100) > 0:
            resultlines100 = list(self.executor.map(self.recognizer100.read, [t.npimg for t in targetdflist100]))
            for i, pred_str in enumerate(resultlines100):
                lineobj = targetdflist100[i]
                lineobj.pred_str = pred_str
                # Long line splitting logic (v1.2.1)
                if len(pred_str) >= 98 and lineobj.npimg.shape[0] < lineobj.npimg.shape[1]:
                    baseimg = lineobj.npimg
                    # Split into two halves
                    tmplineobj_1 = RecogLine(npimg=baseimg[:, :baseimg.shape[1]//2, :], idx=lineobj.idx, pred_char_cnt=100)
                    tmplineobj_2 = RecogLine(npimg=baseimg[:, baseimg.shape[1]//2:, :], idx=lineobj.idx, pred_char_cnt=100)
                    targetdflist200.append(tmplineobj_1)
                    targetdflist200.append(tmplineobj_2)
                else:
                    targetdflistall.append(lineobj)

        # Level 4: Extremely long lines (Split and recognized by PARSEQ-100)
        if len(targetdflist200) > 0:
            resultlines200 = list(self.executor.map(self.recognizer100.read, [t.npimg for t in targetdflist200]))
            for i in range(0, len(targetdflist200) - 1, 2):
                idx_orig = targetdflist200[i].idx
                combined_str = resultlines200[i] + resultlines200[i+1]
                lineobj = RecogLine(npimg=None, idx=idx_orig, pred_char_cnt=100, pred_str=combined_str)
                targetdflistall.append(lineobj)
                    
        # Re-sort results to original order
        targetdflistall = sorted(targetdflistall)
        return [t.pred_str for t in targetdflistall]

    def ocr(
        self,
        pil_image: Image.Image,
        img_name: str = "image.jpg",
        enable_character_inclusion: Optional[bool] = None,
        enable_rich_markdown: Optional[bool] = None,
        enable_table_structure: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Main OCR pipeline.
        1. Layout Detection
        2. XML Representation
        3. Reading Order
        4. Recognition
        """
        img = np.array(pil_image.convert('RGB'))
        img_h, img_w = img.shape[:2]
        
        # 1. Detection
        detections = self.detector.detect(img)
        classeslist = list(self.detector.classes.values())
        
        # Prepare data for NDL-style XML conversion
        resultobj = [
            {0: []},
            {i: [] for i in range(17)}
        ]
        for det in detections:
            xmin, ymin, xmax, ymax = det["box"]
            conf = det["confidence"]
            char_count = det.get("pred_char_count", 100.0)  # v1.2.1 uses char_count in resultobj
            markdown_style = det.get("pred_markdown_style", 0.0)  # v2.0.0 で追加された Markdown スタイル予測値
            if det["class_index"] == 0:
                resultobj[0][0].append([xmin, ymin, xmax, ymax])
            # v2.0.0 に合わせて markdown_style も渡す
            resultobj[1][det["class_index"]].append([xmin, ymin, xmax, ymax, conf, char_count, markdown_style])
            
        # Security: Sanitize img_name to prevent XML injection
        safe_img_name = "".join(c for c in img_name if c.isalnum() or c in "._- ")

        # 2. Convert Detections to XML string and then parse with defusedxml (Security)
        xmlstr = convert_to_xml_string3(img_w, img_h, safe_img_name, classeslist, resultobj)
        xmlstr = "<OCRDATASET>" + xmlstr + "</OCRDATASET>"
        root = ET.fromstring(xmlstr)

        # 3. Reading Order Analysis (modifies XML tree in-place)
        eval_xml(root, logger=None)
        
        # Extract line images based on logical reading order from XML
        alllineobj = []
        lines = root.findall(".//LINE")
        
        tatelinecnt = 0
        alllinecnt = 0
        
        for idx, lineobj in enumerate(lines):
            xmin = int(lineobj.get("X"))
            ymin = int(lineobj.get("Y"))
            line_w = int(lineobj.get("WIDTH"))
            line_h = int(lineobj.get("HEIGHT"))
            try:
                pred_char_cnt = float(lineobj.get("PRED_CHAR_CNT"))
            except (ValueError, TypeError):
                pred_char_cnt = 100.0
            
            if line_h > line_w:
                tatelinecnt += 1
            alllinecnt += 1
            
            lineimg = img[ymin:ymin+line_h, xmin:xmin+line_w, :]
            alllineobj.append(RecogLine(lineimg, idx, pred_char_cnt))

        # Fallback: if XY-Cut fails to find lines but we have detections
        if len(alllineobj) == 0 and len(detections) > 0:
            page = root.find("PAGE")
            for idx, det in enumerate(detections):
                xmin, ymin, xmax, ymax = det["box"]
                line_w = int(xmax - xmin)
                line_h = int(ymax - ymin)
                if line_w > 0 and line_h > 0:
                    line_elem = SubElement(page, "LINE")
                    c_idx = int(det["class_index"])
                    type_name = classeslist[c_idx] if c_idx < len(classeslist) else "本文"
                    line_elem.set("TYPE", type_name)
                    line_elem.set("X", str(int(xmin)))
                    line_elem.set("Y", str(int(ymin)))
                    line_elem.set("WIDTH", str(line_w))
                    line_elem.set("HEIGHT", str(line_h))
                    line_elem.set("CONF", f"{det['confidence']:0.3f}")
                    pred_char_cnt = det.get("pred_char_count", 100.0)
                    line_elem.set("PRED_CHAR_CNT", f"{pred_char_cnt:0.3f}")
                    pred_markdown_style = det.get("pred_markdown_style", 0.0)
                    line_elem.set("PRED_MARKDOWN", f"{pred_markdown_style:0.0f}")
                    if line_h > line_w:
                        tatelinecnt += 1
                    alllinecnt += 1
                    lineimg = img[int(ymin):int(ymax), int(xmin):int(xmax), :]
                    alllineobj.append(RecogLine(lineimg, idx, pred_char_cnt))
            lines = root.findall(".//LINE")

        # 4. Recognition (using cascade and thread pool)
        resultlinesall = self._process_cascade(alllineobj, is_cascade=True)

        # 異体字包摂（有効時のみ適用）
        use_char_inclusion = (
            self.enable_character_inclusion
            if enable_character_inclusion is None
            else enable_character_inclusion
        )
        if use_char_inclusion:
            resultlinesall = [apply_character_inclusion(text) for text in resultlinesall]
        
        # v1.2.1 Verticality check (Reverse text order if majority vertical)
        if alllinecnt > 0 and tatelinecnt / alllinecnt > 0.5:
            full_text = "\n".join(resultlinesall[::-1])
        else:
            full_text = "\n".join(resultlinesall)
        
        # Format results into final JSON structure
        resjsonarray = []
        for idx, lineobj in enumerate(lines):
            lineobj.set("STRING", resultlinesall[idx])
            xmin = int(lineobj.get("X"))
            ymin = int(lineobj.get("Y"))
            line_w = int(lineobj.get("WIDTH"))
            line_h = int(lineobj.get("HEIGHT"))
            try:
                conf = float(lineobj.get("CONF"))
            except (ValueError, TypeError):
                conf = 0.0
                
            # XML TYPE -> c_idx (v1.2.1 improvement)
            type_str = lineobj.get("TYPE", "")
            c_idx = classeslist.index(type_str) if type_str in classeslist else 1
            
            jsonobj = {
                "boundingBox": [[xmin, ymin], [xmin, ymin+line_h], [xmin+line_w, ymin+line_h], [xmin+line_w, ymin]],
                "id": idx,
                "isVertical": "true" if line_h > line_w else "false",
                "text": resultlinesall[idx],
                "isTextline": "true",
                "confidence": conf,
                "class_index": c_idx
            }
            resjsonarray.append(jsonobj)
            
        # 表構造化の実行（有効かつモデルロード済みの場合）
        use_table_structure = (
            self.enable_table_structure
            if enable_table_structure is None
            else enable_table_structure
        )
        tables_result: Optional[List[Dict[str, Any]]] = None
        table_markdowns: Optional[List[Dict[str, Any]]] = None
        page_elem = root.find("PAGE")

        if use_table_structure and page_elem is not None:
            table_recognizer = self._ensure_table_recognizer()
            page_xml = ET.tostring(page_elem, encoding="unicode")
            _, raw_tables = infer_tables_from_page(
                img,
                {"page_xml": page_xml, "json_lines": resjsonarray},
                table_recognizer,
            )
            tables_result = []
            table_markdowns = []
            for item in raw_tables:
                cells = list(item.get("cells", []))
                table_md = "" if item.get("error") else cells_to_markdown(cells)
                tables_result.append({
                    "table_index": int(item.get("table_index", 1)),
                    "layout_bbox": [int(v) for v in item.get("layout_bbox", [])],
                    "html": item.get("html"),
                    "markdown": table_md,
                    "num_cells": int(item.get("num_cells", len(cells))),
                })
                table_markdowns.append({
                    "markdown": table_md,
                    "layout_bbox": item.get("layout_bbox"),
                    "table_index": item.get("table_index"),
                    "source_table_indexes": item.get("source_table_indexes"),
                })

        # 構造化 Markdown の生成（有効時のみ適用）
        use_rich_markdown = (
            self.enable_rich_markdown
            if enable_rich_markdown is None
            else enable_rich_markdown
        )
        markdown_output = full_text
        if use_rich_markdown and page_elem is not None:
            markdown_styles = _normalize_markdown_styles(
                getattr(self.detector, "markdown_styles", None)
            )
            rich_md = build_markdown_text_from_page(
                page_elem,
                table_markdowns=table_markdowns,
                markdown_styles=markdown_styles,
            )
            if rich_md:
                markdown_output = rich_md

        response: Dict[str, Any] = {
            "text": full_text,
            "markdown": markdown_output,
            "lines": resjsonarray,
            "img_info": {
                "width": img_w,
                "height": img_h,
                "name": img_name
            }
        }
        if tables_result is not None:
            response["tables"] = tables_result
        return response

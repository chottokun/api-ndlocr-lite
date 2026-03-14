import sys
import os
import numpy as np
from PIL import Image
from defusedxml import ElementTree as ET
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
    ):
        """
        Initializes the engine with model paths and detection thresholds.
        Loads ONNX models into memory.
        """
        self.device = device
        
        # Default paths pointing into the ndlocr-lite submodule
        base_dir = SUBMODULE_SRC
        self.det_weights = det_weights or str(base_dir / "model" / "deim-s-1024x1024.onnx")
        self.det_classes = det_classes or str(base_dir / "config" / "ndl.yaml")
        self.rec_weights = rec_weights or str(base_dir / "model" / "parseq-ndl-16x768-100-tiny-165epoch-tegaki2.onnx")
        self.rec_weights30 = rec_weights30 or str(base_dir / "model" / "parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx")
        self.rec_weights50 = rec_weights50 or str(base_dir / "model" / "parseq-ndl-16x384-50-tiny-146epoch-tegaki2.onnx")
        self.rec_classes = rec_classes or str(base_dir / "config" / "NDLmoji.yaml")
        
        self.det_score_threshold = det_score_threshold
        self.det_conf_threshold = det_conf_threshold
        self.det_iou_threshold = det_iou_threshold

        self.detector = None
        self.recognizer100 = None
        self.recognizer30 = None
        self.recognizer50 = None
        
        # ThreadPoolExecutor for parallelizing character recognition across lines
        self.executor = ThreadPoolExecutor(
            max_workers=os.cpu_count() or 4,
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

    def _get_recognizer(self, weights_path: str):
        """Helper to initialize a PARSEQ recognizer with a specific weight file."""
        with open(self.rec_classes, encoding="utf-8") as f:
            charobj = safe_load(f)
        charlist = list(charobj["model"]["charset_train"])
        return PARSEQ(model_path=weights_path, charlist=charlist, device=self.device)

    def shutdown(self):
        """Shuts down the internal thread pool."""
        self.executor.shutdown()

    def _process_cascade(self, alllineobj: List[RecogLine], is_cascade: bool = True) -> List[str]:
        """
        Recognition cascading strategy.
        1. Routes lines to PARSEQ-30, 50, or 100 based on 'pred_char_cnt' from the detector.
        2. If a smaller model yields a result longer than its training length, it cascades to the next larger model.
        This optimizes for speed without sacrificing accuracy on long lines.
        """
        targetdflist30 = []
        targetdflist50 = []
        targetdflist100 = []
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

        # Level 3: PARSEQ-100 (Highest capacity, longest lines)
        if len(targetdflist100) > 0:
            resultlines100 = list(self.executor.map(self.recognizer100.read, [t.npimg for t in targetdflist100]))
            for i, pred_str in enumerate(resultlines100):
                lineobj = targetdflist100[i]
                lineobj.pred_str = pred_str
                targetdflistall.append(lineobj)
                    
        # Re-sort results to original order
        targetdflistall = sorted(targetdflistall)
        return [t.pred_str for t in targetdflistall]

    def ocr(self, pil_image: Image.Image, img_name: str = "image.jpg") -> Dict[str, Any]:
        """
        Main OCR pipeline.
        1. Layout Detection: Locates lines, blocks, etc.
        2. XML Representation: Converts detections to NDL-style XML.
        3. Reading Order: Uses XY-Cut algorithm to determine logical reading sequence.
        4. Recognition: Runs character recognition on extracted line images.
        """
        img = np.array(pil_image.convert('RGB'))
        img_h, img_w = img.shape[:2]
        
        # 1. Detection
        detections = self.detector.detect(img)
        classeslist = list(self.detector.classes.values())
        
        # Prepare data for NDL-style XML conversion
        resultobj = [dict(), dict()]
        resultobj[0][0] = list()
        for i in range(17):
            resultobj[1][i] = []
        for det in detections:
            xmin, ymin, xmax, ymax = det["box"]
            conf = det["confidence"]
            if det["class_index"] == 0:
                resultobj[0][0].append([xmin, ymin, xmax, ymax])
            resultobj[1][det["class_index"]].append([xmin, ymin, xmax, ymax, conf])
            
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
        
        for idx, lineobj in enumerate(lines):
            xmin = int(lineobj.get("X"))
            ymin = int(lineobj.get("Y"))
            line_w = int(lineobj.get("WIDTH"))
            line_h = int(lineobj.get("HEIGHT"))
            try:
                pred_char_cnt = float(lineobj.get("PRED_CHAR_CNT"))
            except (ValueError, TypeError):
                pred_char_cnt = 100.0
            
            lineimg = img[ymin:ymin+line_h, xmin:xmin+line_w, :]
            alllineobj.append(RecogLine(lineimg, idx, pred_char_cnt))

        # Fallback: if XY-Cut fails to find lines but we have detections, use raw detections
        if len(alllineobj) == 0 and len(detections) > 0:
            page = root.find("PAGE")
            for idx, det in enumerate(detections):
                xmin, ymin, xmax, ymax = det["box"]
                line_w = int(xmax - xmin)
                line_h = int(ymax - ymin)
                if line_w > 0 and line_h > 0:
                    line_elem = ET.SubElement(page, "LINE")
                    line_elem.set("TYPE", "本文")
                    line_elem.set("X", str(int(xmin)))
                    line_elem.set("Y", str(int(ymin)))
                    line_elem.set("WIDTH", str(line_w))
                    line_elem.set("HEIGHT", str(line_h))
                    line_elem.set("CONF", f"{det['confidence']:0.3f}")
                    pred_char_cnt = det.get("pred_char_count", 100.0)
                    line_elem.set("PRED_CHAR_CNT", f"{pred_char_cnt:0.3f}")
                    lineimg = img[int(ymin):int(ymax), int(xmin):int(xmax), :]
                    alllineobj.append(RecogLine(lineimg, idx, pred_char_cnt))
            lines = root.findall(".//LINE")

        # 4. Recognition (using cascade and thread pool)
        resultlinesall = self._process_cascade(alllineobj, is_cascade=True)
        
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
            jsonobj = {
                "boundingBox": [[xmin, ymin], [xmin, ymin+line_h], [xmin+line_w, ymin+line_h], [xmin+line_w, ymin]],
                "id": idx,
                "text": resultlinesall[idx],
                "confidence": conf
            }
            resjsonarray.append(jsonobj)
            
        full_text = "\n".join(resultlinesall)

        return {
            "text": full_text,
            "lines": resjsonarray,
            "img_info": {
                "width": img_w,
                "height": img_h,
                "name": img_name
            }
        }

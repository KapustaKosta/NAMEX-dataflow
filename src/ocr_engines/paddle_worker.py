import json
import os
import sys

# Set Paddle environment flags to fix oneDNN/PIR runtime crash on Windows CPU
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_use_onednn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("MKLDNN_DISABLE_WORKSPACE", "1")

def main():
    try:
        input_data = sys.stdin.read()
        request = json.loads(input_data)
        
        lang = request.get("lang", "ru")
        use_gpu = request.get("use_gpu", False)
        images = request.get("images", [])
        
        device = "gpu:0" if use_gpu else "cpu"

        # Now import PaddleOCR after env vars are set
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            print(json.dumps({"error": "PaddleOCR not installed"}))
            sys.exit(1)

        # Initialize engine
        paddle_ocr = PaddleOCR(lang=lang, device=device)

        results = []
        for img_info in images:
            page_num = img_info["page"]
            image_path = img_info["path"]
            
            ocr_result = paddle_ocr.ocr(str(image_path))
            
            text_lines = []
            if ocr_result:
                # PaddleOCR 3.x result is typically a list of lists of lines
                # Each line is [ [[x,y], [x,y], [x,y], [x,y]], ("text", confidence) ]
                for page_res in ocr_result:
                    if not page_res:
                        continue
                    for line in page_res:
                        if len(line) >= 2:
                            text_data = line[1]
                            text = text_data[0] if isinstance(text_data, tuple) else str(text_data)
                            confidence = float(text_data[1]) if isinstance(text_data, tuple) and len(text_data) > 1 else 0.9
                            text_lines.append({"text": text, "confidence": confidence})

            results.append({"page": page_num, "lines": text_lines})
        
        print(json.dumps({"success": True, "results": results}))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()

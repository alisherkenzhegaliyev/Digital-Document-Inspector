import uuid
import zipfile
import io
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.utils.pdf_tools import pdf_bytes_to_images, images_to_pdf
from app.services.document_inspector import DocumentInspector
from app.config import MODEL_CONFIGS

router = APIRouter()

# Load all models once globally with cropper parameters
# Use CPU by default, but will use CUDA if available
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
inspector = DocumentInspector(
    MODEL_CONFIGS,
    device=device,
    imgsz=1280  # 'Slight zoom' effect from cropper
)

STATIC_DIR = Path("static/annotated")
STATIC_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/analyze")
async def analyze(pdf_file: UploadFile = File(...)):
    # Validate input type
    if not pdf_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a PDF")

    pdf_bytes = await pdf_file.read()

    # Convert PDF → list of PIL images
    try:
        pages = pdf_bytes_to_images(pdf_bytes)
    except Exception as e:
        error_msg = str(e)
        if "exceeds limit" in error_msg or "decompression bomb" in error_msg:
            raise HTTPException(
                status_code=400, 
                detail="PDF contains very large images. Please try a lower resolution PDF or split it into smaller files."
            )
        raise HTTPException(status_code=500, detail=f"PDF parsing error: {error_msg}")

    # Create a unique job directory
    job_id = uuid.uuid4().hex
    job_dir = STATIC_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Reset statistics for this job
    inspector.reset_statistics()

    output = {
        "job_id": job_id,
        "pages": []
    }

    annotated_images = []   # <-- NEW: store annotated page images for PDF export

    # Process each page
    global_ann_index = 1  # Track annotation number across all pages
    for idx, page_img in enumerate(pages):
        detections, annotated_img = inspector.detect_image(page_img)
        annotated_images.append(annotated_img)

        page_width, page_height = page_img.size

        # Save annotated JPG
        filename = f"page_{idx + 1}.jpg"
        filepath = job_dir / filename
        annotated_img.save(filepath)

        # Format detection structure properly
        formatted_detections = []
        for det in detections:
            formatted_detections.append({
                "category": det["class"],
                "confidence": det["confidence"],
                "bbox": {
                    "x": det["bbox"][0],
                    "y": det["bbox"][1],
                    "width": det["bbox"][2] - det["bbox"][0],
                    "height": det["bbox"][3] - det["bbox"][1]
                }
            })

        output["pages"].append({
            "page_index": idx + 1,
            "page_size": {
                "width": page_width,
                "height": page_height
            },
            "detections": formatted_detections,
            "annotated_image_url": f"/static/annotated/{job_id}/{filename}"
        })

    # === NEW: Generate annotated PDF ===
    annotated_pdf_path = job_dir / "annotated.pdf"
    images_to_pdf(annotated_images, annotated_pdf_path)

    output["annotated_pdf_url"] = f"/static/annotated/{job_id}/annotated.pdf"
    
    # Build parent JSON structure from the already-processed pages (wrapper)
    pdf_name = pdf_file.filename or "document.pdf"
    parent_json = {pdf_name: {}}
    
    global_ann_counter = 1  # Track annotation number across all pages for parent JSON
    for page in output["pages"]:
        page_key = f"page_{page['page_index']}"
        parent_json[pdf_name][page_key] = {
            "annotations": [],
            "page_size": page["page_size"]
        }
        
        # Convert detections to annotation format
        for detection in page["detections"]:
            ann_key = f"annotation_{global_ann_counter}"
            global_ann_counter += 1
            bbox = detection["bbox"]
            area = bbox["width"] * bbox["height"]
            
            annotation_entry = {
                ann_key: {
                    "category": detection["category"],
                    "bbox": bbox,
                    "area": float(area)
                }
            }
            parent_json[pdf_name][page_key]["annotations"].append(annotation_entry)
    
    output["result"] = parent_json
    
    # Add statistics from cropper functionality
    stats = inspector.get_statistics()
    output["statistics"] = stats

    return JSONResponse(output)



@router.post("/batch-analyze")
async def batch_analyze(zip_file: UploadFile = File(...)):
    if not zip_file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a ZIP archive")

    zip_bytes = await zip_file.read()

    try:
        zip_data = zipfile.ZipFile(io.BytesIO(zip_bytes))
        # Try to decode filenames with UTF-8
        zip_data.filename = zip_file.filename
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ZIP file: {e}")

    job_id = uuid.uuid4().hex
    parent_json = {}
    files_processed_count = 0
    
    # Reset statistics for this batch job
    inspector.reset_statistics()

    # Get PDF files and decode filenames properly
    pdf_files = []
    for name in zip_data.namelist():
        if name.startswith("__MACOSX") or name.startswith("._"):
            continue
        if name.lower().endswith(".pdf"):
            # Try to decode filename properly
            try:
                # Some ZIP tools encode filenames incorrectly, try CP437 first then UTF-8
                decoded_name = name.encode('cp437').decode('utf-8')
            except (UnicodeDecodeError, UnicodeEncodeError):
                decoded_name = name
            pdf_files.append((name, decoded_name))
    
    if not pdf_files:
        raise HTTPException(status_code=400, detail="ZIP contains no PDF files")

    # Global annotation counter across ALL PDFs in the batch
    global_ann_index = 1

    for original_name, display_name in pdf_files:
        try:
            pdf_bytes = zip_data.read(original_name)
        except:
            continue

        # Convert PDF to images
        try:
            pages = pdf_bytes_to_images(pdf_bytes)
        except Exception as e:
            error_msg = str(e)
            if "exceeds limit" in error_msg or "decompression bomb" in error_msg:
                parent_json[display_name] = {"error": "PDF contains very large images and cannot be processed"}
            else:
                parent_json[display_name] = {"error": f"PDF parsing failed: {error_msg}"}
            continue

        parent_json[display_name] = {}
        files_processed_count += 1  # Count successfully processed files

        for page_index, page_img in enumerate(pages, start=1):
            detections, annotated_img = inspector.detect_image(page_img)
            w, h = page_img.size

            page_key = f"page_{page_index}"
            parent_json[display_name][page_key] = {
                "annotations": [],
                "page_size": { "width": w, "height": h }
            }

            # Add each detection
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                width = x2 - x1
                height = y2 - y1
                area = width * height

                ann_key = f"annotation_{global_ann_index}"
                global_ann_index += 1

                # Build annotation entry
                annotation_entry = {
                    ann_key: {
                        "category": det["class"],
                        "bbox": {
                            "x": x1,
                            "y": y1,
                            "width": width,
                            "height": height
                        },
                        "area": float(area)
                    }
                }

                parent_json[display_name][page_key]["annotations"].append(annotation_entry)
    
    # Add statistics from cropper functionality
    stats = inspector.get_statistics()

    return JSONResponse({
        "job_id": job_id,
        "files_processed": files_processed_count,
        "result": parent_json,
        "statistics": stats
    })

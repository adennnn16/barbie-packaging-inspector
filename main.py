import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import base64
import io

app = FastAPI(title="MATTEL Barbie Packaging Inspector API")

# Setup template and static directory
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Store current session master image in memory (for single conveyor line setup)
MASTER_IMAGE_GRAY = None
MASTER_IMAGE_COLOR = None

def align_images(im1, im2):
    """
    Align im2 (live image) to im1 (master image) using ORB feature matching.
    Handles minor camera alignment shifts or position variations on the conveyor.
    """
    try:
        MAX_FEATURES = 1000
        GOOD_MATCH_PERCENT = 0.15

        # Convert to grayscale
        im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY) if len(im1.shape) == 3 else im1
        im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY) if len(im2.shape) == 3 else im2

        # Detect ORB features and compute descriptors
        orb = cv2.ORB_create(MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

        if descriptors1 is None or descriptors2 is None:
            return im2

        # Match features
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        # Sort matches by score
        matches = sorted(matches, key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        if len(matches) < 4:
            return im2

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

        if h is None:
            return im2

        # Use homography to warp live image to align with master image
        height, width = im1.shape[:2]
        im2_aligned = cv2.warpPerspective(im2, h, (width, height))

        return im2_aligned
    except Exception as e:
        return im2

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/upload-master")
async def upload_master(file: UploadFile = File(...)):
    global MASTER_IMAGE_GRAY, MASTER_IMAGE_COLOR
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Resize to standard processing size (600x600)
    MASTER_IMAGE_COLOR = cv2.resize(img, (600, 600))
    MASTER_IMAGE_GRAY = cv2.cvtColor(MASTER_IMAGE_COLOR, cv2.COLOR_BGR2GRAY)
    MASTER_IMAGE_GRAY = cv2.GaussianBlur(MASTER_IMAGE_GRAY, (5, 5), 0)

    return JSONResponse({
        "status": "success",
        "message": "Master image uploaded and calibrated successfully."
    })

@app.post("/api/scan-packaging")
async def scan_packaging(file: UploadFile = File(...)):
    global MASTER_IMAGE_GRAY, MASTER_IMAGE_COLOR

    if MASTER_IMAGE_GRAY is None:
        raise HTTPException(status_code=400, detail="Master reference image has not been set yet!")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    live_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if live_img is None:
        raise HTTPException(status_code=400, detail="Invalid live scan image")

    # 1. Resize live image to match master size
    live_img = cv2.resize(live_img, (600, 600))

    # 2. Homography alignment
    aligned_live = align_images(MASTER_IMAGE_COLOR, live_img)

    # 3. Grayscale & Blur
    live_gray = cv2.cvtColor(aligned_live, cv2.COLOR_BGR2GRAY)
    live_gray = cv2.GaussianBlur(live_gray, (5, 5), 0)

    # 4. Image Subtraction / Difference Calculation
    diff = cv2.absdiff(MASTER_IMAGE_GRAY, live_gray)
    
    # 5. Thresholding difference
    _, thresh = cv2.threshold(diff, 45, 255, cv2.THRESH_BINARY)

    # Morphological dilation to close gaps in missing regions
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    # 6. Find Contours of missing / altered items
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated_img = aligned_live.copy()
    missing_count = 0
    missing_areas = []

    # Filter out noise (contour area < 150 pixels)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 180:  # Threshold for small barbie accessories (shoes, comb, etc.)
            missing_count += 1
            x, y, w, h = cv2.boundingRect(cnt)
            missing_areas.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": float(area)})

            # Draw Neon Bounding Box for Missing Item
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 0, 255), 3) # Red Box
            cv2.putText(annotated_img, "MISSING ITEM", (x, max(15, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Convert annotated image back to base64 for frontend display
    _, buffer = cv2.imencode('.jpg', annotated_img)
    base64_img = base64.b64encode(buffer).decode('utf-8')

    is_complete = (missing_count == 0)

    return JSONResponse({
        "status": "COMPLETE" if is_complete else "MISSING_ITEM",
        "is_complete": is_complete,
        "missing_count": missing_count,
        "missing_areas": missing_areas,
        "annotated_image": f"data:image/jpeg;base64,{base64_img}"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

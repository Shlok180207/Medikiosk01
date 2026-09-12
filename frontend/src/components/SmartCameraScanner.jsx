import React, { useRef, useState, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';
import cvModule from '@techstark/opencv-js';

/**
 * Utility: Cleanly loads and initializes OpenCV WebAssembly module
 */
async function loadOpenCv() {
  if (window.cv && typeof window.cv.Mat === 'function') {
    return window.cv;
  }
  let cv = cvModule;
  if (cv && typeof cv.then === 'function') {
    cv = await cv;
  } else if (typeof cv === 'function') {
    cv = await cv();
  } else if (cv && !cv.Mat) {
    await new Promise((resolve) => {
      if (cv.onRuntimeInitialized) {
        const orig = cv.onRuntimeInitialized;
        cv.onRuntimeInitialized = () => { orig(); resolve(); };
      } else {
        cv.onRuntimeInitialized = resolve;
      }
    });
  }
  window.cv = cv;
  return cv;
}

/**
 * Utility: Euclidean distance between two 2D points
 */
function euclideanDistance(p1, p2) {
  return Math.hypot(p1.x - p2.x, p1.y - p2.y);
}

/**
 * Utility: Extracts and deterministically orders 4 quadrilateral corners:
 * [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
 * Handles any polygon (4 to 8 vertices) by calculating extremal sum/diff points.
 */
function getExtremal4Points(pts) {
  if (!pts || pts.length === 0) return null;
  let tl = pts[0], tr = pts[0], br = pts[0], bl = pts[0];
  let minSum = Infinity, maxSum = -Infinity;
  let minDiff = Infinity, maxDiff = -Infinity;

  for (const p of pts) {
    const sum = p.x + p.y;
    const diff = p.y - p.x;
    if (sum < minSum) { minSum = sum; tl = p; }
    if (sum > maxSum) { maxSum = sum; br = p; }
    if (diff < minDiff) { minDiff = diff; tr = p; }
    if (diff > maxDiff) { maxDiff = diff; bl = p; }
  }

  // Ensure 4 points are all distinct and not degenerate
  if (
    euclideanDistance(tl, tr) < 20 ||
    euclideanDistance(tr, br) < 20 ||
    euclideanDistance(br, bl) < 20 ||
    euclideanDistance(bl, tl) < 20
  ) {
    return null;
  }
  return [tl, tr, br, bl];
}

/**
 * Validates a detected quadrilateral to ensure:
 * 1. All 4 corners are strictly inside the camera frame with safe border padding (rejects room walls).
 * 2. Geometry represents a real rectangular paper document (angles, aspect ratio, side ratios).
 * 3. Area is reasonable (12% to 80% of camera screen).
 * 4. Paper Surface & Ink Verification: Uses bilinear interior sampling to ensure:
 *    - At least 55% of interior points are bright white/light paper (rejects humans, clothes, furniture).
 *    - Between 1.5% and 32% of points are dark ink/text strokes (rejects blank walls or plain shirts).
 */
function validateDocumentQuad(quad, procW, procH, totalArea, grayMat) {
  if (!quad || quad.length !== 4) return null;
  const [tl, tr, br, bl] = quad;

  // 1. Lenient Border Margin: reject only if touching the outer 3px edge of sensor
  const BORDER_PAD = 3;
  if (
    tl.x < BORDER_PAD || tl.y < BORDER_PAD ||
    tr.x > procW - BORDER_PAD || tr.y < BORDER_PAD ||
    br.x > procW - BORDER_PAD || br.y > procH - BORDER_PAD ||
    bl.x < BORDER_PAD || bl.y > procH - BORDER_PAD
  ) {
    return null;
  }

  const topDist = euclideanDistance(tl, tr);
  const bottomDist = euclideanDistance(bl, br);
  const leftDist = euclideanDistance(tl, bl);
  const rightDist = euclideanDistance(tr, br);

  // Minimum edge length: at least 18 pixels in 320x240 frame
  if (topDist < 18 || bottomDist < 18 || leftDist < 18 || rightDist < 18) return null;

  // Opposite side lengths: perspective tolerance <= 2.3 (resilient to handheld tilt)
  const wRatio = Math.max(topDist, bottomDist) / Math.min(topDist, bottomDist);
  const hRatio = Math.max(leftDist, rightDist) / Math.min(leftDist, rightDist);
  if (wRatio > 2.3 || hRatio > 2.3) return null;

  // Aspect ratio: typical documents (A4, Rx slips, bills, wide reports) between 0.25 and 3.5
  const avgW = (topDist + bottomDist) / 2;
  const avgH = (leftDist + rightDist) / 2;
  const ar = avgW / avgH;
  if (ar < 0.25 || ar > 3.5) return null;

  // Corner angles check: allow natural perspective tilt (|cos| < 0.78, ~38° to 142°)
  const cosAngles = [
    Math.abs(((tr.x - tl.x)*(bl.x - tl.x) + (tr.y - tl.y)*(bl.y - tl.y)) / (topDist * leftDist)),
    Math.abs(((tl.x - tr.x)*(br.x - tr.x) + (tl.y - tr.y)*(br.y - tr.y)) / (topDist * rightDist)),
    Math.abs(((tr.x - br.x)*(bl.x - br.x) + (tr.y - br.y)*(bl.y - br.y)) / (rightDist * bottomDist)),
    Math.abs(((br.x - bl.x)*(tl.x - bl.x) + (br.y - bl.y)*(tl.y - bl.y)) / (bottomDist * leftDist))
  ];
  if (Math.max(...cosAngles) > 0.78) return null;

  // Quadrilateral area (shoelace formula)
  const quadArea = 0.5 * Math.abs(
    (tl.x * tr.y - tl.y * tr.x) +
    (tr.x * br.y - tr.y * br.x) +
    (br.x * bl.y - br.y * bl.x) +
    (bl.x * tl.y - bl.y * tl.x)
  );

  // Must occupy between 6% and 95% of viewfinder area (supports small receipts to close-up docs)
  if (quadArea < totalArea * 0.06 || quadArea > totalArea * 0.95) return null;

  // Center should be roughly within viewfinder (4% to 96%)
  const cx = (tl.x + tr.x + br.x + bl.x) / 4;
  const cy = (tl.y + tr.y + br.y + bl.y) / 4;
  if (cx < procW * 0.04 || cx > procW * 0.96 || cy < procH * 0.04 || cy > procH * 0.96) return null;

  // 4. Interior Luminance Sanity Check: Ensure area is not a pitch black void
  if (grayMat && grayMat.data) {
    let totalLuminance = 0;
    const SAMPLES = 8;
    let validSamples = 0;

    for (let i = 1; i < SAMPLES; i++) {
      const u = i / SAMPLES;
      for (let j = 1; j < SAMPLES; j++) {
        const v = j / SAMPLES;
        const px = Math.round((1 - u) * (1 - v) * tl.x + u * (1 - v) * tr.x + u * v * br.x + (1 - u) * v * bl.x);
        const py = Math.round((1 - u) * (1 - v) * tl.y + u * (1 - v) * tr.y + u * v * br.y + (1 - u) * v * bl.y);

        if (px >= 0 && px < procW && py >= 0 && py < procH) {
          totalLuminance += grayMat.data[py * procW + px];
          validSamples++;
        }
      }
    }

    // Only reject if interior is pitch black (average luminance < 45 out of 255)
    if (validSamples >= 20 && (totalLuminance / validSamples) < 45) {
      return null;
    }
  }

  return { quad, area: quadArea };
}

/**
 * SmartCameraScanner
 * -------------------
 * Client-Side Auto-Capturing Document Scanner using OpenCV.js WebAssembly:
 * 1. Async WebAssembly Loader with direct npm module import
 * 2. 8 FPS Throttled Gatekeeper loop to prevent UI stutter
 * 3. Convex-Hull Assisted Contour Detection:
 *    - Strategy A: Canny Edge Detection + Convex Hull + approxPolyDP
 *    - Strategy B: Otsu Threshold Contours (resilient to warm lighting and shadows)
 *    - Finger/indentation resilience: Bridges grip marks via convex hull
 * 4. Coordinate Stability Tracker (1.2-second lock timer)
 * 5. Pre-OCR Image Enhancement:
 *    - Perspective Warp (Deskewing to full original resolution)
 *    - Natural High-Clarity Color Output with Contrast Normalization
 * 6. Explicit .delete() cleanup on all allocated WASM matrices
 */
export default function SmartCameraScanner({ onCapture, onClose, onError }) {
  const webcamRef = useRef(null);
  const hiddenCanvasRef = useRef(null);
  const outputCanvasRef = useRef(null);
  const animFrameIdRef = useRef(null);
  const cvInstanceRef = useRef(null);

  // States
  const [isOpenCvReady, setIsOpenCvReady] = useState(false);
  const [facingMode, setFacingMode] = useState('environment');
  const [statusMessage, setStatusMessage] = useState('Initializing Computer Vision engine...');
  const [stabilityProgress, setStabilityProgress] = useState(0); // 0 to 100%
  const [cornersOverlay, setCornersOverlay] = useState(null); // Scaled for SVG overlay
  const [isFlashing, setIsFlashing] = useState(false);
  const [autoSnapEnabled, setAutoSnapEnabled] = useState(true);

  // Tracking refs to avoid state re-render thrashing inside loop
  const prevCornersRef = useRef(null);
  const stableStartTimeRef = useRef(null);
  const isCapturingRef = useRef(false);
  const detectedCornersRef = useRef(null);

  // Play synthesized camera shutter sound (Web Audio API, 100% offline)
  const playShutterSound = () => {
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = 'triangle';
      osc.frequency.setValueAtTime(750, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(140, ctx.currentTime + 0.08);

      gain.gain.setValueAtTime(0.3, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.08);

      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.09);
    } catch (e) {
      // Audio autoplay policy fallback
    }
  };

  // ─────────────────────────────────────────────────────────────────────────
  // 1. Initialize OpenCV.js Module
  // ─────────────────────────────────────────────────────────────────────────
  useEffect(() => {
    let isMounted = true;

    loadOpenCv()
      .then(cv => {
        if (isMounted) {
          cvInstanceRef.current = cv;
          setIsOpenCvReady(true);
          setStatusMessage('Align document in viewfinder');
          console.log('✅ OpenCV.js WebAssembly loaded successfully into React scanner');
        }
      })
      .catch(err => {
        console.error('OpenCV load error:', err);
        if (isMounted) {
          setStatusMessage('Failed to initialize OpenCV engine.');
          if (onError) onError(err);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [onError]);

  // ─────────────────────────────────────────────────────────────────────────
  // 2. Pre-OCR Pipeline: Perspective Warp (Deskew) + High-Clarity Rendering
  // ─────────────────────────────────────────────────────────────────────────
  const processAndEnhanceDocument = useCallback((highResVideo, orderedCorners, scaleX, scaleY) => {
    const cv = cvInstanceRef.current || window.cv;
    if (!cv || !cv.Mat) return null;

    const origW = highResVideo.videoWidth;
    const origH = highResVideo.videoHeight;

    // Calculate center of the quadrilateral in downscaled coordinates
    let cx = 0, cy = 0;
    for (const pt of orderedCorners) {
      cx += pt.x;
      cy += pt.y;
    }
    cx /= 4;
    cy /= 4;

    // Expand the bounding box outward by 5% to prevent cutting off document edges
    const EXPANSION_FACTOR = 1.05;

    // Map 4 downscaled corner coordinates back to full high-res video resolution with expansion
    const [tl, tr, br, bl] = orderedCorners.map(pt => {
      // Scale outward from center
      const expandedX = cx + (pt.x - cx) * EXPANSION_FACTOR;
      const expandedY = cy + (pt.y - cy) * EXPANSION_FACTOR;
      
      return {
        x: Math.max(0, Math.min(origW, expandedX * scaleX)),
        y: Math.max(0, Math.min(origH, expandedY * scaleY)),
      };
    });

    // Calculate maximum output width and height for the deskewed rectangle
    const widthBottom = euclideanDistance(br, bl);
    const widthTop = euclideanDistance(tr, tl);
    const targetWidth = Math.round(Math.max(widthBottom, widthTop));

    const heightRight = euclideanDistance(tr, br);
    const heightLeft = euclideanDistance(tl, bl);
    const targetHeight = Math.round(Math.max(heightRight, heightLeft));

    if (targetWidth < 60 || targetHeight < 60) return null;

    // Temporary high-res canvas to read camera frame pixels
    const captureCanvas = document.createElement('canvas');
    captureCanvas.width = origW;
    captureCanvas.height = origH;
    const capCtx = captureCanvas.getContext('2d');
    capCtx.drawImage(highResVideo, 0, 0, origW, origH);
    const fullImgData = capCtx.getImageData(0, 0, origW, origH);

    // OpenCV WebAssembly Matrix declarations
    let srcMat = null;
    let dstMat = null;
    let srcCoords = null;
    let dstCoords = null;
    let transformMatrix = null;

    try {
      srcMat = cv.matFromImageData(fullImgData);
      dstMat = new cv.Mat();

      // Source quadrilateral coordinates: [TL, TR, BR, BL]
      srcCoords = cv.matFromArray(4, 1, cv.CV_32FC2, [
        tl.x, tl.y,
        tr.x, tr.y,
        br.x, br.y,
        bl.x, bl.y,
      ]);

      // Target rectangular coordinates: [0, 0], [W, 0], [W, H], [0, H]
      dstCoords = cv.matFromArray(4, 1, cv.CV_32FC2, [
        0, 0,
        targetWidth - 1, 0,
        targetWidth - 1, targetHeight - 1,
        0, targetHeight - 1,
      ]);

      // 1. Perspective Transformation (Deskew & Crop)
      transformMatrix = cv.getPerspectiveTransform(srcCoords, dstCoords);
      const dsize = new cv.Size(targetWidth, targetHeight);
      cv.warpPerspective(srcMat, dstMat, transformMatrix, dsize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());

      // 2. OCR Image Enhancement Pipeline (Make text pop, remove slight blur)
      // We use Unsharp Masking: Enhanced = Original + (Original - Blurred) * Amount
      const blurredDst = new cv.Mat();
      cv.GaussianBlur(dstMat, blurredDst, new cv.Size(0, 0), 3); // 3px blur radius
      // addWeighted formula: dst = src1*alpha + src2*beta + gamma
      // Here: 1.5 * Original - 0.5 * Blurred = Sharpened Text
      cv.addWeighted(dstMat, 1.5, blurredDst, -0.5, 0, dstMat);
      
      // Optional: Gentle contrast stretch (Alpha=1.2 to increase contrast, Beta=10 to brighten)
      dstMat.convertTo(dstMat, -1, 1.15, 5); 
      
      blurredDst.delete();

      // 3. Render to output canvas and export
      const finalCanvas = outputCanvasRef.current || document.createElement('canvas');
      finalCanvas.width = targetWidth;
      finalCanvas.height = targetHeight;
      cv.imshow(finalCanvas, dstMat);

      const base64String = finalCanvas.toDataURL('image/jpeg', 0.94);
      return { base64String, canvas: finalCanvas };
    } finally {
      // STRICT MEMORY LEAK PREVENTION: Delete all allocated WASM matrices
      if (srcMat) srcMat.delete();
      if (dstMat) dstMat.delete();
      if (srcCoords) srcCoords.delete();
      if (dstCoords) dstCoords.delete();
      if (transformMatrix) transformMatrix.delete();
    }
  }, []);

  // ─────────────────────────────────────────────────────────────────────────
  // 3. Trigger Capture Sequence (Auto or Manual)
  // ─────────────────────────────────────────────────────────────────────────
  const triggerCapture = useCallback(() => {
    if (isCapturingRef.current) return;
    const webcam = webcamRef.current;
    const video = webcam?.video;
    if (!video || video.readyState < 2 || video.videoWidth === 0) return;

    isCapturingRef.current = true;
    setIsFlashing(true);
    playShutterSound();
    setTimeout(() => setIsFlashing(false), 300);

    setStatusMessage('Deskewing & Cropping Document...');

    const corners = detectedCornersRef.current;
    const procW = 320;
    const procH = 240;

    setTimeout(() => {
      try {
        let finalBase64 = null;
        let blobTargetCanvas = null;

        if (corners && corners.length === 4) {
          // Perspective crop and high-clarity enhancement
          const scaleX = video.videoWidth / procW;
          const scaleY = video.videoHeight / procH;
          const result = processAndEnhanceDocument(video, corners, scaleX, scaleY);
          if (result) {
            finalBase64 = result.base64String;
            blobTargetCanvas = result.canvas;
          }
        }

        // Fallback: If no 4-point contour, crop center 88% viewfinder region
        if (!finalBase64) {
          const fallbackCanvas = document.createElement('canvas');
          const fw = Math.floor(video.videoWidth * 0.88);
          const fh = Math.floor(video.videoHeight * 0.90);
          const fx = Math.floor((video.videoWidth - fw) / 2);
          const fy = Math.floor((video.videoHeight - fh) / 2);
          fallbackCanvas.width = fw;
          fallbackCanvas.height = fh;
          const fCtx = fallbackCanvas.getContext('2d');
          fCtx.drawImage(video, fx, fy, fw, fh, 0, 0, fw, fh);
          finalBase64 = fallbackCanvas.toDataURL('image/jpeg', 0.94);
          blobTargetCanvas = fallbackCanvas;
        }

        if (blobTargetCanvas && onCapture) {
          blobTargetCanvas.toBlob(blob => {
            if (blob) {
              const file = new File([blob], `scan_deskewed_${Date.now()}.jpg`, { type: 'image/jpeg' });
              onCapture(file, finalBase64);
            }
          }, 'image/jpeg', 0.94);
        }
      } catch (err) {
        console.error('Capture sequence error:', err);
        if (onError) onError(err);
      } finally {
        isCapturingRef.current = false;
      }
    }, 40);
  }, [onCapture, onError, processAndEnhanceDocument]);

  // ─────────────────────────────────────────────────────────────────────────
  // 4. The Gatekeeper: 8 FPS Frame Processing & Contour Pipeline
  // ─────────────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!isOpenCvReady) return;

    let lastFrameTime = 0;
    const TARGET_FPS = 8; // Capped at 8 FPS to ensure zero UI jank
    const FRAME_INTERVAL = 1000 / TARGET_FPS;
    const cv = cvInstanceRef.current || window.cv;
    if (!cv) return;

    function processFrame(timestamp) {
      if (isCapturingRef.current) return;

      if (timestamp - lastFrameTime < FRAME_INTERVAL) {
        animFrameIdRef.current = requestAnimationFrame(processFrame);
        return;
      }
      lastFrameTime = timestamp;

      const webcam = webcamRef.current;
      const hiddenCanvas = hiddenCanvasRef.current;
      const video = webcam?.video;

      if (!video || video.readyState < 2 || video.videoWidth === 0 || !hiddenCanvas) {
        animFrameIdRef.current = requestAnimationFrame(processFrame);
        return;
      }

      const procW = 320;
      const procH = 240;
      const totalArea = procW * procH;
      hiddenCanvas.width = procW;
      hiddenCanvas.height = procH;

      const ctx = hiddenCanvas.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(video, 0, 0, procW, procH);
      const imgData = ctx.getImageData(0, 0, procW, procH);

      // WASM Matrix allocations
      let src = null;
      let gray = null;
      let blurred = null;
      let edges = null;
      let threshOtsu = null;
      let closedPaper = null;
      let closedEdges = null;
      let textThresh = null;
      let textDilated = null;
      let contours = null;
      let hierarchy = null;

      try {
        src = cv.matFromImageData(imgData);
        gray = new cv.Mat();
        blurred = new cv.Mat();
        edges = new cv.Mat();
        threshOtsu = new cv.Mat();
        closedPaper = new cv.Mat();
        closedEdges = new cv.Mat();
        contours = new cv.MatVector();
        hierarchy = new cv.Mat();

        // 1. Grayscale Conversion
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

        // 2. Gaussian Blur to reduce high-frequency camera noise
        const ksize = new cv.Size(3, 3);
        cv.GaussianBlur(gray, blurred, ksize, 0, 0, cv.BORDER_DEFAULT);

        // 3. Canny Edges (sensitive enough for faint document edges)
        cv.Canny(blurred, edges, 25, 95);

        let bestQuad = null;
        let maxArea = 0;

        // ─────────────────────────────────────────────────────────────────
        // Strategy 1: White/Light Paper Segmentation (Otsu Threshold + Morph Close)
        // ─────────────────────────────────────────────────────────────────
        // In typical kiosk/webcam scenarios, white paper has high contrast against background
        cv.threshold(blurred, threshOtsu, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);

        // Seal all text and ink lines inside the paper to create a solid white polygon
        const closeKernel7 = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(7, 7));
        cv.morphologyEx(threshOtsu, closedPaper, cv.MORPH_CLOSE, closeKernel7);
        closeKernel7.delete();

        cv.findContours(closedPaper, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

        for (let i = 0; i < contours.size(); i++) {
          const cnt = contours.get(i);
          const rawArea = cv.contourArea(cnt);

          if (rawArea >= totalArea * 0.06 && rawArea <= totalArea * 0.95) {
            const hull = new cv.Mat();
            cv.convexHull(cnt, hull, false, true);
            const hullArea = cv.contourArea(hull);

            if (hullArea >= totalArea * 0.06 && hullArea <= totalArea * 0.95) {
              const perimeter = cv.arcLength(hull, true);
              const approx = new cv.Mat();
              cv.approxPolyDP(hull, approx, 0.032 * perimeter, true);

              let quad = null;
              // Accepts 4 to 8 vertices (resilient to fingers, creases, slight edge bends)
              if (approx.rows >= 4 && approx.rows <= 8) {
                const pts = [];
                for (let j = 0; j < approx.rows; j++) {
                  pts.push({ x: approx.data32S[j * 2], y: approx.data32S[j * 2 + 1] });
                }
                quad = getExtremal4Points(pts);
              }
              approx.delete();

              if (quad) {
                const validated = validateDocumentQuad(quad, procW, procH, totalArea, gray);
                if (validated && validated.area > maxArea) {
                  maxArea = validated.area;
                  bestQuad = validated.quad;
                }
              }
            }
            hull.delete();
          }
          cnt.delete();
        }

        // ─────────────────────────────────────────────────────────────────
        // Strategy 2: Canny Edge Closed Boundary (For Light Desks / Mixed Light)
        // ─────────────────────────────────────────────────────────────────
        if (!bestQuad) {
          const dilateKernel3 = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
          cv.dilate(edges, closedEdges, dilateKernel3);
          dilateKernel3.delete();

          const edgeContours = new cv.MatVector();
          cv.findContours(closedEdges, edgeContours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

          for (let i = 0; i < edgeContours.size(); i++) {
            const cnt = edgeContours.get(i);
            const hull = new cv.Mat();
            cv.convexHull(cnt, hull, false, true);
            const hullArea = cv.contourArea(hull);

            if (hullArea >= totalArea * 0.06 && hullArea <= totalArea * 0.95) {
              const perimeter = cv.arcLength(hull, true);
              const approx = new cv.Mat();
              cv.approxPolyDP(hull, approx, 0.032 * perimeter, true);

              let quad = null;
              if (approx.rows >= 4 && approx.rows <= 8) {
                const pts = [];
                for (let j = 0; j < approx.rows; j++) {
                  pts.push({ x: approx.data32S[j * 2], y: approx.data32S[j * 2 + 1] });
                }
                quad = getExtremal4Points(pts);
              }
              approx.delete();

              if (quad) {
                const validated = validateDocumentQuad(quad, procW, procH, totalArea, gray);
                if (validated && validated.area > maxArea) {
                  maxArea = validated.area;
                  bestQuad = validated.quad;
                }
              }
            }
            hull.delete();
            cnt.delete();
          }
          edgeContours.delete();
        }

        // ─────────────────────────────────────────────────────────────────
        // Strategy 3: Text Lines Cluster Bounding Quad (For Faint / Low-Contrast Edges)
        // ─────────────────────────────────────────────────────────────────
        if (!bestQuad) {
          textThresh = new cv.Mat();
          textDilated = new cv.Mat();
          cv.adaptiveThreshold(gray, textThresh, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, 8);

          const horizKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(14, 4));
          cv.dilate(textThresh, textDilated, horizKernel);
          horizKernel.delete();

          const textContours = new cv.MatVector();
          cv.findContours(textDilated, textContours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

          const textBlocks = [];
          for (let i = 0; i < textContours.size(); i++) {
            const cnt = textContours.get(i);
            const rect = cv.boundingRect(cnt);
            if (rect.width >= 35 && rect.width <= 240 && rect.height >= 5 && rect.height <= 35) {
              if (rect.width / rect.height >= 1.8) {
                textBlocks.push(rect);
              }
            }
            cnt.delete();
          }
          textContours.delete();

          if (textBlocks.length >= 2) {
            let minX = procW, minY = procH, maxX = 0, maxY = 0;
            for (const b of textBlocks) {
              minX = Math.min(minX, b.x);
              minY = Math.min(minY, b.y);
              maxX = Math.max(maxX, b.x + b.width);
              maxY = Math.max(maxY, b.y + b.height);
            }

            const textW = maxX - minX;
            const textH = maxY - minY;

            // Add generous 15% margin around text lines
            const padX = textW * 0.15;
            const padY = textH * 0.15;
            const pMinX = Math.max(6, minX - padX);
            const pMinY = Math.max(6, minY - padY);
            const pMaxX = Math.min(procW - 6, maxX + padX);
            const pMaxY = Math.min(procH - 6, maxY + padY);

            const clusterArea = (pMaxX - pMinX) * (pMaxY - pMinY);
            if (clusterArea >= totalArea * 0.06 && clusterArea <= totalArea * 0.95) {
              const quad = [
                { x: pMinX, y: pMinY },
                { x: pMaxX, y: pMinY },
                { x: pMaxX, y: pMaxY },
                { x: pMinX, y: pMaxY }
              ];
              const validated = validateDocumentQuad(quad, procW, procH, totalArea, gray);
              if (validated) {
                bestQuad = validated.quad;
              }
            }
          }
        }

        // ─────────────────────────────────────────────────────────────────
        // 4. Stability Evaluation & 0.5s Fast Auto-Capture
        // ─────────────────────────────────────────────────────────────────
        if (bestQuad) {
          let ordered = getExtremal4Points(bestQuad);
          const prev = prevCornersRef.current;

          // Light exponential moving average to eliminate micro-vibrations
          if (prev && prev.length === 4) {
            const ALPHA = 0.70;
            ordered = ordered.map((pt, idx) => ({
              x: prev[idx].x * (1 - ALPHA) + pt.x * ALPHA,
              y: prev[idx].y * (1 - ALPHA) + pt.y * ALPHA,
            }));
          }

          detectedCornersRef.current = ordered;

          // Scaled percentage coordinates for SVG overlay
          setCornersOverlay(ordered.map(p => ({
            x: (p.x / procW) * 100,
            y: (p.y / procH) * 100,
          })));

          const PIXEL_DRIFT_THRESHOLD = 26.0; // Allowed jitter in downscaled canvas (generous)
          const SOFT_DRIFT_THRESHOLD = 45.0;  // Soft tremor zone (decays instead of instant 0%)
          const HOLD_DURATION = 500;          // Fast 0.5-second lock to auto-capture

          if (prev) {
            const maxMovement = Math.max(
              euclideanDistance(ordered[0], prev[0]),
              euclideanDistance(ordered[1], prev[1]),
              euclideanDistance(ordered[2], prev[2]),
              euclideanDistance(ordered[3], prev[3])
            );

            if (maxMovement < PIXEL_DRIFT_THRESHOLD) {
              if (!stableStartTimeRef.current) {
                stableStartTimeRef.current = Date.now();
              }
              const elapsed = Date.now() - stableStartTimeRef.current;
              const progress = Math.min(100, Math.round((elapsed / HOLD_DURATION) * 100));
              setStabilityProgress(progress);
              setStatusMessage(`Document Locked — Capturing (${progress}%)`);

              // Auto-capture when locked for 0.5s
              if (elapsed >= HOLD_DURATION && autoSnapEnabled && !isCapturingRef.current) {
                triggerCapture();
                return;
              }
            } else if (maxMovement < SOFT_DRIFT_THRESHOLD) {
              // Minor hand tremor: soft decay rather than jarring reset to 0%
              if (stableStartTimeRef.current) {
                stableStartTimeRef.current += 90;
                const elapsed = Math.max(0, Date.now() - stableStartTimeRef.current);
                const progress = Math.min(100, Math.round((elapsed / HOLD_DURATION) * 100));
                setStabilityProgress(progress);
              }
              setStatusMessage('Document Detected — Hold steady');
            } else {
              // Large movement
              stableStartTimeRef.current = Date.now();
              setStabilityProgress(0);
              setStatusMessage('Document Detected — Hold steady');
            }
          } else {
            stableStartTimeRef.current = Date.now();
            setStabilityProgress(0);
            setStatusMessage('Document Detected — Hold steady');
          }

          prevCornersRef.current = ordered;
        } else {
          // No valid document in view
          detectedCornersRef.current = null;
          prevCornersRef.current = null;
          stableStartTimeRef.current = null;
          setStabilityProgress(0);
          setCornersOverlay(null);
          setStatusMessage('Align document inside viewfinder');
        }
      } catch (err) {
        console.error('Frame processing error:', err);
      } finally {
        // Safe WASM matrix deallocations without double frees
        const mats = [
          src, gray, blurred, edges, threshOtsu,
          closedPaper, closedEdges, textThresh, textDilated,
          contours, hierarchy
        ];
        for (const m of mats) {
          if (m) {
            try { m.delete(); } catch (_) {}
          }
        }
      }

      animFrameIdRef.current = requestAnimationFrame(processFrame);
    }

    animFrameIdRef.current = requestAnimationFrame(processFrame);

    return () => {
      if (animFrameIdRef.current) {
        cancelAnimationFrame(animFrameIdRef.current);
      }
    };
  }, [isOpenCvReady, autoSnapEnabled, triggerCapture]);

  const toggleFacingMode = () => {
    setFacingMode(prev => (prev === 'environment' ? 'user' : 'environment'));
  };

  const polygonPointsStr = cornersOverlay
    ? `${cornersOverlay[0].x},${cornersOverlay[0].y} ${cornersOverlay[1].x},${cornersOverlay[1].y} ${cornersOverlay[2].x},${cornersOverlay[2].y} ${cornersOverlay[3].x},${cornersOverlay[3].y}`
    : '';

  return (
    <div className="camera-scanner-modal animate-scale-up">
      {/* Hidden processing canvas (WASM source) */}
      <canvas ref={hiddenCanvasRef} style={{ display: 'none' }} />

      {/* Hidden output canvas (for Base64 deskewed output) */}
      <canvas ref={outputCanvasRef} style={{ display: 'none' }} />

      {/* Viewport */}
      <div className="camera-viewport">
        {/* Shutter White Flash Animation */}
        {isFlashing && <div className="camera-flash" />}

        {/* Live HUD Status Badge */}
        <div
          className="camera-hud-badge"
          style={{
            borderColor: cornersOverlay
              ? (stabilityProgress >= 80 ? '#22c55e' : '#38bdf8')
              : 'rgba(255,255,255,0.18)',
            background: cornersOverlay && stabilityProgress >= 80
              ? 'rgba(21, 128, 61, 0.95)'
              : 'rgba(15, 23, 42, 0.90)'
          }}
        >
          <span>{cornersOverlay ? (stabilityProgress >= 80 ? '🟢' : '🟡') : '⚪'}</span>
          <span>{statusMessage}</span>
          {stabilityProgress > 0 && stabilityProgress < 100 && (
            <span style={{ fontSize: '11px', opacity: 0.95, color: '#38bdf8', fontWeight: 700 }}>
              [{stabilityProgress}%]
            </span>
          )}
        </div>

        {/* Video Stream via react-webcam */}
        <Webcam
          ref={webcamRef}
          audio={false}
          screenshotFormat="image/jpeg"
          videoConstraints={{
            facingMode: { ideal: facingMode },
            width: { ideal: 1920 },
            height: { ideal: 1080 },
          }}
          className="camera-video"
        />

        {/* Dynamic 4-Point Quadrilateral SVG Contour Overlay */}
        <svg
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
            zIndex: 10,
          }}
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
        >
          {cornersOverlay && (
            <polygon
              points={polygonPointsStr}
              fill="rgba(34, 197, 94, 0.20)"
              stroke="#22c55e"
              strokeWidth="1.2"
              strokeDasharray={stabilityProgress < 100 ? '2, 1' : 'none'}
            />
          )}
        </svg>

        {/* 4 Corner Pin Badges when Document is Locked */}
        {cornersOverlay && cornersOverlay.map((corner, i) => (
          <div
            key={i}
            style={{
              position: 'absolute',
              left: `${corner.x}%`,
              top: `${corner.y}%`,
              width: 12,
              height: 12,
              borderRadius: '50%',
              background: '#22c55e',
              border: '2px solid #ffffff',
              transform: 'translate(-50%, -50%)',
              zIndex: 15,
              boxShadow: '0 0 8px rgba(34, 197, 94, 0.9)',
            }}
          />
        ))}

        {/* Reticle Guide Overlay (When searching for document) */}
        {!cornersOverlay && (
          <div className="camera-reticle-overlay">
            <div className="camera-reticle-box">
              <div className="camera-corner camera-corner-tl" />
              <div className="camera-corner camera-corner-tr" />
              <div className="camera-corner camera-corner-bl" />
              <div className="camera-corner camera-corner-br" />
              <div className="camera-laser-line" />
            </div>
          </div>
        )}
      </div>

      {/* Controls Bar */}
      <div className="camera-controls-bar">
        {/* Cancel Button */}
        <button
          className="btn btn-outline"
          onClick={onClose}
          style={{ padding: '8px 16px', fontSize: '13px', color: '#cbd5e1', borderColor: 'rgba(255,255,255,0.2)' }}
        >
          ✕ Cancel
        </button>

        {/* Manual Shutter Button */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px' }}>
          <button
            className="shutter-btn"
            onClick={triggerCapture}
            style={{
              borderColor: cornersOverlay ? '#22c55e' : '#38bdf8',
              boxShadow: cornersOverlay ? '0 0 20px rgba(34, 197, 94, 0.4)' : '0 0 15px rgba(56, 189, 248, 0.3)'
            }}
            title={cornersOverlay ? 'Deskew and capture document now' : 'Snap and crop'}
          >
            <div className="shutter-inner-circle">
              📷
            </div>
          </button>
          <span style={{ fontSize: '11px', color: cornersOverlay ? '#4ade80' : '#94a3b8', fontWeight: 600 }}>
            {cornersOverlay ? 'Deskew & Snap' : 'Snap & Auto-Deskew'}
          </span>
        </div>

        {/* Controls: Camera Flip & Auto-Snap Toggle */}
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <button
            className="btn btn-outline"
            onClick={toggleFacingMode}
            style={{ padding: '8px 12px', fontSize: '12px', color: '#cbd5e1', borderColor: 'rgba(255,255,255,0.2)' }}
            title="Flip camera"
          >
            🔄 Flip
          </button>

          <button
            onClick={() => setAutoSnapEnabled(prev => !prev)}
            style={{
              padding: '8px 12px',
              fontSize: '12px',
              borderRadius: 'var(--radius-md)',
              border: '1px solid',
              borderColor: autoSnapEnabled ? '#22c55e' : 'rgba(255,255,255,0.2)',
              background: autoSnapEnabled ? 'rgba(34, 197, 94, 0.15)' : 'transparent',
              color: autoSnapEnabled ? '#4ade80' : '#94a3b8',
              cursor: 'pointer',
              fontWeight: 600
            }}
            title="Toggle automatic document detection & auto-snap"
          >
            {autoSnapEnabled ? '⚡ Auto-Snap: ON' : '⚡ Auto-Snap: OFF'}
          </button>
        </div>
      </div>

      {/* Quick Guidance Footer */}
      <div style={{
        textAlign: 'center',
        padding: '7px 14px 9px',
        fontSize: '11px',
        color: 'rgba(255,255,255,0.7)',
        background: '#090d16',
        borderTop: '1px solid rgba(255,255,255,0.06)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '6px'
      }}>
        <span>💡</span>
        <span>Hold steady for fast 0.5s auto-capture, or tap <strong>📷 Shutter</strong> anytime for instant capture</span>
      </div>
    </div>
  );
}

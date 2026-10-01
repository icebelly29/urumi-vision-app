# Urumi Vision App — Flowcharts

To view or export these diagrams, paste each code block into https://mermaid.live

---

## 1. High-Level Diagram

Shows the overall system flow across the Desktop Receiver and Mobile App.

```mermaid
flowchart TD
    subgraph DESKTOP["Desktop — test_receiver.html"]
        A([Start]) --> B[Create PeerJS Connection ID]
        B --> C[Generate QR Code with Connection URL]
        C --> D{Wait for Mobile Connection}
        D -->|Connected| E[Show Mobile Ready status]
        E --> F{Wait for Payload}
        F -->|PROCESSING_COMPLETE| G[Display Processed Image]
        G --> H[Render Layered SVG]
        H --> I[Show Overlay View]
        I --> Z([End])
    end

    subgraph MOBILE["Mobile — index.html / app.js"]
        M1([User opens URL from QR]) --> M2[Connect to Desktop via WebRTC PeerJS]
        M2 --> M3["Send MOBILE_APP_READY signal"]
        M3 --> M4[Wait for User to Upload Image]
        M4 --> M5{File Type?}
        M5 -->|SVG file| M6[Send SVG directly to Desktop]
        M5 -->|Image file JPG or PNG| M7{Magic Pen Toggle ON?}
        M7 -->|NO| M8["runPipeline — processor.process"]
        M7 -->|YES| M9["processor.flatten — Normalize Perspective"]
        M9 --> M10[Open Lasso Tool — User draws selection mask]
        M10 --> M11["runPipeline — processor.processFlattened with mask"]
        M8 --> M12[Send Payload via WebRTC — SVG + Image + Metadata]
        M11 --> M12
        M6 --> M12
        M12 --> M13[Show Success Status on Mobile]
    end

    C -.->|User scans QR code| M1
    M2 -.->|WebRTC connection established| D
    M3 -.->|MOBILE_APP_READY data received| E
    M12 -.->|PROCESSING_COMPLETE data received| F
```

---

## 2. Low-Level Diagram

Shows the internal data flow inside the image processing pipeline.

```mermaid
flowchart TD
    A([Image file selected by user]) --> B[Draw image to hidden HTML Canvas]
    B --> C{Magic Pen Enabled?}

    C -->|YES| D["processor.flatten(canvas)"]
    D --> D1["WarpEngine.normalizePerspective(src, null)"]
    D1 --> D2{4 ArUco markers detected?}
    D2 -->|NO| ERR1[ERROR — Could not detect markers]
    D2 -->|YES| D3[Apply Perspective Warp — Output: warpedMat]
    D3 --> D4[Render warpedMat to new HTML Canvas]
    D4 --> D5[Open Lasso Modal — Display flattenedCanvas to user]

    D5 --> D6{User selection mode?}
    D6 -->|Polygon mode| D7[User clicks points on image to form shape]
    D6 -->|Rectangle mode| D8[User drags a rectangle region]
    D7 --> D9[User clicks Confirm]
    D8 --> D9
    D9 --> D10[Generate binary maskCanvas — White = selected, Black = excluded]
    D10 --> D11["processor.processFlattened(flattenedCanvas, maskCanvas)"]

    D11 --> E1["cv.imread flattenedCanvas to warpedMat"]
    D11 --> E2["cv.imread maskCanvas to warpedMask, cvtColor to GRAY"]
    E1 & E2 --> F

    C -->|NO| G["processor.process(canvas, null)"]
    G --> G1["WarpEngine.normalizePerspective(src, null)"]
    G1 --> G2{4 ArUco markers detected?}
    G2 -->|NO| ERR2[ERROR — Could not detect markers]
    G2 -->|YES| G3[Apply Perspective Warp — Output: warpedMat with null mask]
    G3 --> F

    F["InkExtractor.extractColorPaths — warpedMat, frameConfig, borderSizeMm, mask"]
    F --> F1[Isolate Blue pixels — thru_cut layer paths]
    F --> F2[Isolate Red pixels — score layer paths]
    F --> F3[Isolate Green pixels — crease layer paths]
    F1 & F2 & F3 --> H

    H["SvgGenerator.generateLayeredSvg — layersData, width, height, frameConfig"]
    H --> H1[Output: svgContent string]
    H --> H2[Output: metaContent — dots_per_mm and physical dimensions]
    H1 & H2 --> I[Convert warpedMat to JPEG Data URL]
    I --> J["Return result object — svg, image, meta"]

    J --> K{WebRTC connection open?}
    K -->|YES| K1["communicator.sendPayload() — Send PROCESSING_COMPLETE over PeerJS"]
    K -->|NO| K2["LocalStorage Bridge fallback — Write to urumi_payload key, then remove"]

    K1 --> L[Desktop Receiver displays result]
    K2 --> L2[Desktop Receiver picks up via storage event listener]
    L & L2 --> Z([Pipeline Complete])
```

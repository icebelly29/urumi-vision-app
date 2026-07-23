# Integration Guide: WebRTC Vision Scanner

The mobile web application utilizes WebRTC (via PeerJS) for peer-to-peer data transfer, removing the need for any backend server while ensuring the payload is securely delivered to the Main UI.

## Setup Instructions:
1. **Install PeerJS in your project:**
   ```html
   <!-- If using HTML -->
   <script src="https://cdn.jsdelivr.net/npm/peerjs@1.3.2/dist/peerjs.min.js"></script>
   ```
   Or via npm: `npm install peerjs`

2. **Initialize WebRTC in the Main UI:**

   ```javascript
   // Generate a unique ID for this session or let PeerJS assign one
   const mainUiPeer = new Peer('main-ui-scanner-12345'); 

   mainUiPeer.on('open', (id) => {
       console.log('Main UI listening on ID: ' + id);
       
       // Generate QR Code containing this URL:
       // https://YOUR-GITHUB-PAGES-URL/mobile-app/index.html?peerId=main-ui-scanner-12345
       generateQRCode(`https://YOUR_DOMAIN/mobile-app/index.html?peerId=${id}`);
   });

   mainUiPeer.on('connection', (conn) => {
       console.log('Mobile device connected!');

       conn.on('data', (data) => {
           if (data.type === 'MOBILE_APP_READY') {
               console.log('Mobile app is open and ready to capture.');
           }
           else if (data.type === 'PROCESSING_COMPLETE') {
               console.log('Received payload from mobile app!');
               
               const finalSvgString = data.payload.svg;
               const previewImageUrl = data.payload.image; // Base64 Data URL

               // Inject SVG into your editor, parse layers, etc.
               handleIncomingDrawing(finalSvgString, previewImageUrl);
           }
       });
   });
   ```

## Payload Structure

When the capture and processing pipeline completes on the mobile device, you will receive the `PROCESSING_COMPLETE` event. The SVG string adheres strictly to the required multi-layer standard:

```xml
<svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="1000" height="750" viewBox="0 0 1000 750">
  <g id="layer_bed_frame">
    <rect x="0" y="0" width="1000" height="750" fill="none" stroke="#FF00FF" stroke-width="2"/>
  </g>
  <g id="layer_thru_cut">
    <!-- Paths matching black -->
  </g>
  <g id="layer_score">
    <!-- Paths matching blue -->
  </g>
  <g id="layer_crease">
    <!-- Paths matching red -->
  </g>
</svg>
```

You can now parse this SVG and map it directly into the UrumiCutter workspace!

class SvgGenerator {
    static generateLayeredSvg(layersData, width, height, frameConfig, cardboardPolyBed = null) {
        let physWidth = 775.0;
        let physHeight = 1065.0;

        if (frameConfig) {
            physWidth = frameConfig.cropPhysW || 775.0;
            physHeight = frameConfig.cropPhysH || 1065.0;
        }

        const dotsPerMm = width / physWidth;
        const toMm = (px) => (px / dotsPerMm);
        const toMmX = (px) => toMm(px).toFixed(3);
        const toMmY = (px) => toMm(px).toFixed(3);

        let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${physWidth} ${physHeight}" width="100%" style="display:block;">\n`;
        svg += `  <style>path, circle, ellipse, polygon, polyline, line, rect { fill: none; stroke-width: 1; stroke-linejoin: round; stroke-linecap: round; }</style>\n`;

        // 1. Bed Reference Layer
        svg += `  <g id="bed" data-role="reference">\n`;
        svg += `    <rect x="0" y="0" width="${physWidth}" height="${physHeight}" fill="none" stroke="#888" stroke-width="0.6" stroke-dasharray="4,4" data-role="reference"/>\n`;
        if (cardboardPolyBed && cardboardPolyBed.length >= 3) {
            const cbPts = cardboardPolyBed.map(pt => `${toMmX(pt[0])},${toMmY(pt[1])}`).join(' ');
            svg += `    <polygon points="${cbPts}" fill="none" stroke="#888" stroke-width="0.6" stroke-dasharray="4,4" data-role="reference"/>\n`;
        }
        svg += `  </g>\n`;

        // 2. Color Layers in strict order: red, green, blue
        const orderedLayers = [
            { id: 'red', defaultColor: '#e53935', aliases: ['red', 'score'] },
            { id: 'green', defaultColor: '#43a047', aliases: ['green', 'crease'] },
            { id: 'blue', defaultColor: '#1e88e5', aliases: ['blue', 'thru_cut'] }
        ];

        for (const layerSpec of orderedLayers) {
            let paths = [];
            let strokeColor = layerSpec.defaultColor;
            for (const alias of layerSpec.aliases) {
                if (layersData && layersData[alias] && Array.isArray(layersData[alias].paths)) {
                    paths = paths.concat(layersData[alias].paths);
                    if (layersData[alias].color) strokeColor = layersData[alias].color;
                }
            }
            svg += `  <g id="${layerSpec.id}">\n`;
            const commonAttrs = `fill="none" stroke="${strokeColor}" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"`;
            paths.forEach(shape => {
                if (shape.type === 'circle') {
                    svg += `    <circle cx="${toMmX(shape.cx)}" cy="${toMmY(shape.cy)}" r="${toMm(shape.r).toFixed(3)}" ${commonAttrs}/>\n`;
                } else if (shape.type === 'ellipse') {
                    const cxMm = toMmX(shape.cx);
                    const cyMm = toMmY(shape.cy);
                    const rxMm = toMm(shape.rx).toFixed(3);
                    const ryMm = toMm(shape.ry).toFixed(3);
                    const ang = (shape.angle_deg || 0).toFixed(1);
                    svg += `    <ellipse cx="${cxMm}" cy="${cyMm}" rx="${rxMm}" ry="${ryMm}" transform="rotate(${ang} ${cxMm} ${cyMm})" ${commonAttrs}/>\n`;
                } else if (shape.type === 'rect') {
                    const xMm = toMmX(shape.x);
                    const yMm = toMmY(shape.y);
                    const wMm = toMm(shape.width).toFixed(3);
                    const hMm = toMm(shape.height).toFixed(3);
                    const ang = +(shape.angle_deg || 0);
                    if (Math.abs(ang) > 0.5) {
                        const cxMm = (parseFloat(xMm) + 0.5 * parseFloat(wMm)).toFixed(3);
                        const cyMm = (parseFloat(yMm) + 0.5 * parseFloat(hMm)).toFixed(3);
                        svg += `    <rect x="${xMm}" y="${yMm}" width="${wMm}" height="${hMm}" transform="rotate(${ang.toFixed(1)} ${cxMm} ${cyMm})" ${commonAttrs}/>\n`;
                    } else {
                        svg += `    <rect x="${xMm}" y="${yMm}" width="${wMm}" height="${hMm}" ${commonAttrs}/>\n`;
                    }
                } else if (shape.type === 'polygon') {
                    const pts = shape.points.map(pt => `${toMmX(pt[0])},${toMmY(pt[1])}`).join(' ');
                    svg += `    <polygon points="${pts}" ${commonAttrs}/>\n`;
                } else if (shape.type === 'line') {
                    svg += `    <line x1="${toMmX(shape.x1)}" y1="${toMmY(shape.y1)}" x2="${toMmX(shape.x2)}" y2="${toMmY(shape.y2)}" ${commonAttrs}/>\n`;
                } else if (shape.type === 'bezier' && shape.beziers && shape.beziers.length > 0) {
                    const b0 = shape.beziers[0];
                    let d = `M ${toMmX(b0[0][0])},${toMmY(b0[0][1])}`;
                    for (const seg of shape.beziers) {
                        d += ` C ${toMmX(seg[1][0])},${toMmY(seg[1][1])} ${toMmX(seg[2][0])},${toMmY(seg[2][1])} ${toMmX(seg[3][0])},${toMmY(seg[3][1])}`;
                    }
                    svg += `    <path d="${d}" ${commonAttrs}/>\n`;
                } else if (shape.type === 'polyline' || shape.type === 'freehand') {
                    if (!shape.points || shape.points.length < 2) return;
                    const pts = shape.points.map(pt => `${toMmX(pt[0])},${toMmY(pt[1])}`).join(' ');
                    svg += `    <polyline points="${pts}" ${commonAttrs}/>\n`;
                } else if (shape.type === 'path') {
                    if (!shape.points || shape.points.length < 2) return;
                    const d = `M ${shape.points.map(pt => `${toMmX(pt[0])},${toMmY(pt[1])}`).join(' L ')}`;
                    svg += `    <path d="${d}" ${commonAttrs}/>\n`;
                }
            });
            svg += `  </g>\n`;
        }

        svg += `</svg>`;
        return {
            svg: svg,
            meta: {
                dots_per_mm: dotsPerMm,
                physical_width: physWidth,
                physical_height: physHeight
            }
        };
    }
}

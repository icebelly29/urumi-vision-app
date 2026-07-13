class SvgGenerator {
    static generateLayeredSvg(layersData, width, height, frameConfig) {
        let physWidth = 176;
        let physHeight = 256;

        if (frameConfig) {
            physWidth = frameConfig.cropPhysW;
            physHeight = frameConfig.cropPhysH;
        }

        // dotsPerMm: how many image pixels = 1 physical millimeter
        const dotsPerMm = width / physWidth;

        // SVG coordinate system: 1 SVG unit = 1 mm.
        // All path coordinates are divided by dotsPerMm to convert pixels → mm.
        // viewBox matches the physical mm dimensions exactly.
        // SVG uses top-left origin, same as the warped image. No Y-flip needed.
        // Any G-code coordinate flipping should happen in the post-processor.
        const toMm = (px) => (px / dotsPerMm);
        const toMmX = (px) => toMm(px).toFixed(3);
        const toMmY = (px) => toMm(px).toFixed(3);

        let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${physWidth} ${physHeight}" width="100%" style="display:block;">\n`;
        svg += `  <style>path { fill: none; stroke-width: 0.5mm; vector-effect: non-scaling-stroke; stroke-linejoin: round; stroke-linecap: round; }</style>\n`;

        // 1. Bed Frame Layer — the boundary of the physical frame in mm
        svg += `  <g id="layer_bed_frame">\n`;
        svg += `    <rect x="0" y="0" width="${physWidth}" height="${physHeight}" fill="none" stroke="#FF00FF" stroke-width="0.5"/>\n`;
        svg += `  </g>\n`;

        // 2. Data Layers — all coordinates in mm
        for (const [layerId, data] of Object.entries(layersData)) {
            if (data.paths.length === 0) continue;

            svg += `  <g id="${layerId}">\n`;
            data.paths.forEach(shape => {
                if (shape.type === 'circle') {
                    svg += `    <circle cx="${toMmX(shape.cx)}" cy="${toMmY(shape.cy)}" r="${toMm(shape.r).toFixed(3)}" fill="none" stroke="${data.color}" stroke-width="0.5"/>\n`;
                } else if (shape.type === 'polygon') {
                    const pts = shape.points.map(pt => `${toMmX(pt[0])},${toMmY(pt[1])}`).join(' ');
                    svg += `    <polygon points="${pts}" fill="none" stroke="${data.color}" stroke-width="0.5" stroke-linejoin="round"/>\n`;
                } else if (shape.type === 'path') {
                    if (shape.points.length < 2) return;
                    const d = `M ${shape.points.map(pt => `${toMmX(pt[0])},${toMmY(pt[1])}`).join(' L ')}`;
                    svg += `    <path d="${d}" fill="none" stroke="${data.color}" stroke-width="0.5" stroke-linejoin="round" stroke-linecap="round"/>\n`;
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

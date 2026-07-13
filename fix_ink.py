import re

with open('pipeline/InkExtractor.js', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix Adaptive Threshold for Shadows
content = content.replace(
    'cv.adaptiveThreshold(blurred, darkStrokes, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 401, 20);',
    'cv.adaptiveThreshold(blurred, darkStrokes, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 115, 20);'
)

# 2. Fix Color Classification Saturation Threshold
content = content.replace(
    'if (bestS < 35 || bestV > 250)',
    'if (bestS < 20 || bestV > 250)'
)

# 3. Widen Red Hue slightly
content = content.replace(
    'else if (bestH < 10 || bestH >= 165)',
    'else if (bestH < 15 || bestH >= 165)'
)

# 4. Fix Crumbled Paths (Lower epsilon in _simplifyPath call)
content = content.replace(
    'return { type: \'path\', points: this._simplifyPath(this._smoothPath(path, 8), 1.2) };',
    'return { type: \'path\', points: this._simplifyPath(this._smoothPath(path, 15), 0.8) };'
)
content = content.replace(
    'return { type: \'path\', points: this._simplifyPath(smoothed, 2.5) };',
    'return { type: \'path\', points: this._simplifyPath(smoothed, 1.2) };'
)

with open('pipeline/InkExtractor.js', 'w', encoding='utf-8') as f:
    f.write(content)

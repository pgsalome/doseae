# IJROBP / Red Journal Figure Guidelines

Use these rules when creating publication-quality figures for radiation
oncology or medical physics manuscripts.

## General Requirements

- Use final-print sizing, not oversized figures with tiny text.
- Use Arial or Helvetica font.
- Use readable font sizes at final size:
  - Axis labels: 10 pt
  - Tick labels: 8-9 pt
  - Legend: 8-9 pt
  - Panel labels: 12-14 pt bold
  - In-image annotations: 8-10 pt
- Export figures at 600 dpi for plots and 300-600 dpi for medical images.
- Save each figure as both PDF and high-resolution PNG or TIFF.
- Every axis must have a clear label and unit, for example "Time from treatment, months", "Dose, Gy", "Volume, cm^3", or "Survival probability".
- Define all abbreviations in the legend.
- Do not place figure-level titles inside the artwork unless explicitly requested. Panel titles are acceptable when they clarify multi-panel figures. Put the overall explanatory title in the manuscript caption instead.
- Do not rely on color alone. Use different line styles, markers, or hatch patterns to distinguish groups.
- Use no more than six panels per figure.
- Use panel labels A, B, C, etc. in the upper-left corner of each panel.

## Kaplan-Meier / Survival Curves

- Create a Kaplan-Meier plot with a number-at-risk table below the curve.
- Show censored patients on each curve using visible censor marks.
- Use distinct line styles in addition to colors, for example solid versus dashed.
- Label the x-axis as "Time from treatment, months".
- Label the y-axis as "Survival probability" or the specific endpoint, for example "Hearing-loss-free survival probability".
- Include the number at risk at clinically meaningful time points.
- Include log-rank p-value if available.
- Include hazard ratio and 95% confidence interval if available.
- Keep risk table font readable, around 8 pt.

## Multi-Cohort Plots

- Groups must be distinguishable without color alone.
- Use line style, marker shape, or hatch pattern in addition to color.
- Keep legends concise and readable.
- Avoid overcrowding the figure.

## CT / MRI / Dose / Isodose / Contour Figures

- Use grayscale anatomy with high-contrast contours.
- Make isodose lines and contours thick enough to be visible, around 2-3 pt.
- Label relevant structures and isodose levels clearly.
- Define all contour and isodose abbreviations in the legend.
- Apply brightness and contrast adjustments uniformly to the whole image only. Do not selectively enhance individual regions.

## Matplotlib Defaults

Use these default settings unless there is a strong reason to change them:

```python
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 2.2,
    "axes.linewidth": 1.0,
    "savefig.dpi": 600,
})
```

Use these figure sizes:

- Single-column plot: `figsize=(3.5, 3.0)`
- Double-column plot: `figsize=(7.0, 4.5)`
- Kaplan-Meier curve with risk table: `figsize=(7.0, 5.5)`
- Multi-panel figure: `figsize=(7.0, 5.5)` or `figsize=(7.0, 6.0)`, depending on layout

## Code Expectations

New plotting scripts should be modular and should:

1. Load the input data.
2. Create the requested figure.
3. Apply Red Journal-style formatting.
4. Add appropriate axis labels, units, legends, panel labels, and panel titles when useful. Do not add figure-level titles inside the artwork unless explicitly requested.
5. Save the figure as PDF and PNG or TIFF at publication quality.
6. Keep all output filenames explicit and manuscript-ready.

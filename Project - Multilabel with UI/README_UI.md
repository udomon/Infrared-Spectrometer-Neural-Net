# IR spectrum predictor UI

Copy `Spectrum_UI.py` into the existing `Project - Multilabel copy` folder, alongside `Models.py` and `trained_weights.pt` (or one folder below the weights). Then run:

```powershell
python Spectrum_UI.py
```

Use a clean image containing one plotted IR trace. The calibration clicks should be:

1. 4000 cm⁻¹ on the left of the x-axis.
2. 400 cm⁻¹ on the right of the x-axis.
3. 0% transmittance at the bottom of the y-axis.
4. 100% transmittance at the top of the y-axis.

The image automatically expands to the largest size that fits the window. The cyan overlay shows the digitised trace used by the network. If it does not follow the curve, adjust the dark-pixel threshold and predict again.

The UI uses a separate threshold for every functional group from `thresholds.json`. Run the updated training script once to generate the file after validation. Until it exists, the UI clearly indicates that it is using the 0.50 fallback for every group.

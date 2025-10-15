import splitfolders

splitfolders.ratio(
    "training/Dataset",
    output="training/fromcodesplitted",
    seed=42,
    ratio=(0.7, 0.2, 0.1)
)

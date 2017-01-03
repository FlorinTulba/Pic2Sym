## Patches Provider module

[Back to the Appendix](../appendix.md) or jump to the [start page](../../../../ReadMe.md)

-------

![](SupplyOfPatches_classes.jpg)<br>

An original image ***Img*** will be resized to ***ResizedImg***, so that:

- its new size is less or equal to its initial size
- the resulted sides are multiples of the symbol size (read from ***SymSettings***) and thus can be covered by square patches of the size of the glyphs
- the obtained count of patches per axis is not larger than the corresponding value imposed in ***ImgSettings***

The image transformation requires:

- only the normal-size ***Patch***es (when preselection mode is off)
- normal-size and tiny versions of the ***Patch***es (when preselection mode is on)

The ***Patch***es are provided by the ***TransformSupport*** object which is managed by ***PreselManager***. All ***Patch***es are then reused until a new image is loaded.

The configuration file [**res/varConfig.txt**][varConfig] provides the option of transforming less noisy versions of images (based on this option, the approximations can consider only rather pronounced edges within the patches). The mentioned switch is off by default.

-------
[Back to the Appendix](../appendix.md) or jump to the [start page](../../../../ReadMe.md)

[varConfig]:../../../../res/varConfig.txt

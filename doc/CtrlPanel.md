## Description of the Control Panel ##
[Back to start page](../ReadMe.md)

These are the controls explained below:
![](CtrlPanel.jpg)

- Button **Select an Image** prompts the user with the Open File Dialog to choose which image to approximate by symbols
- Button **Transform the Image** performs the approximation if there is an image and the configuration is complete and correct. The results are saved in ***Output*** folder
- Slider **Max horizontal symbols** (*mhs*) limits the width of the grid that approximates the image. Before the transformation, the original is resized to be at most _mhs*fontSize_ pixels wide
- Slider **Max vertical symbols** (*mvs*) limits the height of the grid that approximates the image. Before the transformation, the original is resized to be at most _mvs*fontSize_ pixels tall
- Button **Select a Scalable, preferably also Monospaced Font Family** prompts the user with the Select Font Dialog to choose which font family to use when approximating an image. Monospace fonts have same width, so a more even width distribution - feature useful when approximating patches with random width distributions. Please ignore *Font Size* and *Script* presented by the standard dialog, as *they are configured by the next 2 sliders*. After selection, the ***symbol set window*** will quickly present some of the symbols from the **default encoding** of the chosen font family
- Slider **Encoding** lets the user decide which encoding to use from the selected font family (when there are more available). The *name of the current encoding* appears at the end of the status bar in the ***symbol set window***. Examples of typical encoding names: *UNICODE*, *APPLE ROMAN*, *ADOBE STANDARD*. Sliders provided by *highgui* module from ***OpenCV*** must contain the values 0 and 1, so font families with a single encoding will still appear as offering 2. The application ignores bad encoding requests
- Slider **Font size** allows resizing the selected font using scales beyond the normal sizes exposed by the Select Font Dialog. The valid range is 7..50. Smaller values mean more faithful approximations and more computation time, while larger ones produce coarser results, but faster. Symbols less than 7x7 are really hard to distinguish
- The buttons **Restore defaults for values below** and **Set as defaults the values below** are handy when experimenting with the transformation parameters
- Slider **Hybrid Result** is merely a checkbox:
	- 0 means that the result is the *actual* approximation of the image (from version 1.0), with no cosmeticizing
	- 1 is for *Hybrid Result* - combination of approximated patches with the blurred version of those original patches. Good approximations will be more visible on the result patch, while the rest of less-inspired approximations will be dominated by the blurred patch
- Slider **Structural similarity** (see [this](https://ece.uwaterloo.ca/~z70wang/research/ssim) for details) does contribute to generating *great approximations*, however it incurs a *much longer transformation*
- The sliders **Fit under**, **Fit edge**, **Fit aside** penalize poor approximations (they aim for the *correctness* of the match):
	- **Fit under** will measure the match betweeen the foreground of the symbol and the patch. Normally, this checks an area smaller than the regions used by the next 2 sliders. This setting can _remain smaller than the values of *Fit edge* and *Fit aside*_
	- **Fit edge** assesses how different is the contour of the glyph from the corresponding patch region. This _should be set larger than *Fit under* and *Fit aside*_
	- **Fit aside** evaluates the similarity between the background of the symbol and the corresponding zone from the patch. It involves typically the largest investigated patch region, so it _matters more than *Fit under*_
- Slider **Contrast** penalizes weak foreground-background contrast of the approximations. Sometimes this favors small symbols with high\-contrast while ignoring relevant texture around the considered symbol
- The sliders **Mass Center** and **Direction** help to the *smoothness*, not the accuracy of the result. When viewing the result from a certain distance, it is best that each approximated patch to preserve its direction of flow in the result. So if an original patch seems to be brighter in the top\-right corner, then the approximation should also be a symbol that makes the top\-right corner brighter. Such a gradient-like mechanism was implemented by comparing the mass-centers of the patch and of the candidate symbol, in terms of how far are they apart as distance and as angle
- Slider **Larger Symbols** permits generating *fancier*, but less accurate approximations. It&#39;s just more appealing to see K,S,H instead of commas and quotes
- Slider **Blanks below Threshold** spares the viewer from struggling to read symbols with the difference between foreground and background under a certain threshold, so in the generated result *such barely visible symbols are replaced by blanks*
- Button **About** offers a short description of the application
- Button **Instructions** displays a shorter version of these explanations
- The buttons **Load/Save Settings** ensure rapid setup of various complete scenarios of transformation (apart from the image to approximate).

The _sliders which are set to 0 disable the corresponding aspect_ and reduce the transformation time. When all sliders are on 0, all patches get approximated by the first symbol in the set.

----------
[Back to start page](../ReadMe.md)


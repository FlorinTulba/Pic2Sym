# Pic2Sym #

**Topics:**

1. Description of Pic2Sym application \(*see below*\)
2. [Related work](doc/relatedWork.md)
3. [Example results](doc/results.md)
4. [Testing the Provided Binary](doc/testBinary.md) \- ++Windows 64bits only++
5. [Performance](doc/performance.md)
6. [Conclusions and further directions](doc/conclusions.md)
	[Appendix](doc/appendix.md) \(*various technical details*\)

## Description of the application ##

*Pic2Sym* **approximates images** by a **grid of colored symbols with colored backgrounds**.

It runs on **Windows \(at least version 7; 64 bits\)**. It comes with a *graphical interface* and a *console* tracing relevant events.

The symbols used for approximation are **scalable**, *preferably* **fixed\-width** fonts, like *Consolas*, *Courier New* and *Lucida Console*. See [this](https://en.wikipedia.org/wiki/Samples_of_monospaced_typefaces) for more examples. Scalable fonts allow using virtually any size of the font while preserving its quality. Fixed\-width fonts are more evenly width\-distributed, thus more helpful to approximate random patches.

The resulted grid has **square cells** and *the symbols are preprocessed* to fit such cells by widening them all. Therefore the **output is simply a new image**, and cannot be saved as a char table in html or as a text file, like in similar applications.
Console output is also ruled out, first because there are only a couple of font sizes available and secondly, because consoles usually provide only 16 colors.

- - -

This is how *Pic2Sym* looks like:

| ![](doc/EndOfTransformation.jpg) | while transforming an image |
| -: | :- |
| The **main window** allows **comparing** the *original image* with the *approximation result* using the **Transparency slider** at the bottom of the window.<br>**Magnifying a region** is possible using the toolbar or the mouse wheel. \(Large zoom factors display even the values of the pixels.\)<br>Same window is used to [present mismatches detected during Unit Tests](doc/UnitTesting.md). | ![](doc/MainAfterTransform.jpg) |
| ![](doc/CmapViewer.jpg) | **Symbol set window** displays the **symbols used for the approximation**, one page at a time. The presented glyphs are *already resized to fit in square cells*. *Blanks and exact\-duplicates were removed*, as well.<br>The status bar provides *complete information about these symbols* \(font family, style, size, encoding and total count\).<br>Notice that most symbols on the first rows are not centered after the resize, leaving a large gap to their right. This happens for **non**\-*fixed\-width* fonts, which are **not** *recommended for approximations*. The glyphs in the first image \(presenting the application\) are fixed\-width. |
| The **Control Panel** contains all the necessary controls to **customize and generate approximations for various images**.<br>The user can **display / hide** it using **Ctrl\+P** or the **last tool from the toolbar** from any other window.<br>[Here](doc/CtrlPanel.md) are more explanations about the controls. | ![](doc/CtrlPanelAndInstructions.jpg) |

To **leave the application**, please **activate a window** (except the console and the Control Panel) and **press ESC**. \(This is a limitation of ***highgui*** module from ***OpenCV***.\)

----------

Kindly address any observations, suggestions or questions to me using <***florintulba{at}yahoo{dot}com***>.<br>&copy; 2016 Florin Tulba
